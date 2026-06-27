# ==================== Import Packages ==================== #
import time
import sys
import os 

from typing import Tuple
import numpy as np 
import json 
import math 

import torch
from torch import nn

from PIL import Image

from torchvision.transforms import transforms

import clip

from scipy import spatial

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from torchvision.transforms import functional as F

# ==================== Constant Parameters ==================== #


# ==================== Functions ==================== #
def sliding_window_sum(x : torch.Tensor,
                       kernel : Tuple[int, int]=(16,16),
                       stride      : Tuple[int, int]=(16,16),
                       padding     : Tuple[int, int]=(0,0)):
    """
    Accumulate values inside a sliding window.
    ------------------------------------------------------------
    Args
    x            : 2-D tensor of shape (H, W)
    kernel       : (kh, kw)  window size
    stride       : (sh, sw)  stride
    padding      : (ph, pw)  zero-padding around the input (same as conv padding)

    Returns
    win_sum      : (nH, nW)  sum of elements inside each window
    """

    import torch.nn.functional as F

    if x.dim() != 2:
        raise ValueError(f'Expect a 2-D tensor (H,W), got shape {tuple(x.shape)}')

    # im2col only supports float, cast to float32
    x = x.to(dtype=torch.float32)

    kh, kw = kernel
    sh, sw = stride
    ph, pw = padding

    # Unfold all windows on a 4-D tensor (N=1, C=1, H, W)
    x4d = x.unsqueeze(0).unsqueeze(0)          # -> (1,1,H,W)
    # After unfold: (N, C*kh*kw, L), where L = nH*nW
    patches = F.unfold(x4d,
                       kernel_size=(kh, kw),
                       padding    =(ph, pw),
                       stride     =(sh, sw))
    # Sum over kh*kw to get per-window total
    win_sum = patches.sum(dim=1)               # shape -> (1, L)

    H, W = x.shape
    H_pad, W_pad = H + 2*ph, W + 2*pw
    nH = (H_pad - kh) // sh + 1
    nW = (W_pad - kw) // sw + 1

    win_sum = win_sum.view(nH, nW)

    return win_sum

def input_process_image(img, img_size=512):
    w, h = img.size
    r = w / h 

    if w > h:
        w_new = math.ceil(math.sqrt(img_size * img_size * r))
        h_new = math.ceil(w_new / r)
    else:
        h_new = math.ceil(math.sqrt(img_size * img_size / r))
        w_new = math.ceil(h_new * r)
    h_new = math.ceil(h_new) // 16 * 16
    w_new = math.ceil(w_new) // 16 * 16

    img_resized = img.resize((w_new, h_new))

    return img_resized, img.size
    
def load_image(image):
    
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = image.unsqueeze(0)
        return image
    elif isinstance(image, Image.Image):
        image = F.to_tensor(image.convert("RGB"))
        image = image.unsqueeze(0)
        return image
    elif isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, str):
        image = F.to_tensor(Image.open(image).convert("RGB"))
        image = image.unsqueeze(0)
        return image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
        
def edit_region_soft_iou(gt_mask_uint8: np.ndarray,
                         diff_map: np.ndarray,
                         eps: float = 1e-8) -> float:
    """
    gt_mask_uint8 : H x W, 0/255
    diff_map      : H x W, arbitrary real values (signed allowed); |diff| recommended
    returns       : ERSI score in [0, 1]
    """

    # 1) GT -> {0, 1}
    G = (gt_mask_uint8 >= 255).astype(np.float32)

    # 2) Predicted diff -> [0, 1] soft mask (magnitude only)
    diff = np.abs(diff_map)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + eps)
    P = diff.astype(np.float32)

    # 3) Soft-IoU
    inter = (P * G).sum()
    union = P.sum() + G.sum() - inter
    return float(inter / (union + eps))

def get_sum_activation_score(gt_mask_uint8,
                             diff_map,
                             eps: float = 1e-8,):
    """
    Total activation score (PyTorch).

    Args
    ----
    gt_mask_uint8 : torch.Tensor, uint8 mask in [0, 255] where 255 marks the foreground.
    diff_map      : torch.Tensor, arbitrary-shape diff map (signed allowed).
    eps           : float, small constant to avoid division by zero.
    """

    # |diff|, then softmax over all pixels
    diff = diff_map.abs()
    flat_prob = torch.softmax(diff.flatten(), dim=0)

    # Binary GT mask: 255 -> 1.0, otherwise 0.0
    G = (gt_mask_uint8 >= 255).float()

    # Dot product
    score = torch.dot(flat_prob, G.flatten())

    return score.item() 

def dilate_mask(mask: torch.Tensor, n: int = 1, keep_value: int = 255) -> torch.Tensor:
    """
    Dilate a single-channel binary mask by n pixels in all 8 directions.

    mask        : shape can be [H,W], [1,H,W], [B,H,W] or [B,1,H,W];
                  values may be 0/1 or 0/255.
    n           : number of pixels to dilate (n=1 means expand by one pixel).
    keep_value  : value used to fill foreground in the output (1 or 255).
    """
    import torch.nn.functional as F

    if n <= 0:
        return mask.clone()

    # 1. Normalize to [B,1,H,W] and remember the original dtype/shape
    orig_dtype  = mask.dtype
    orig_shape  = mask.shape
    if mask.dim() == 2:               # [H,W]
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:             # [B,H,W]
        mask = mask.unsqueeze(1)
    elif mask.dim() == 4 and mask.size(1) == 1:
        pass                          # already [B,1,H,W]
    else:
        raise ValueError("mask must be single-channel with shape [H,W] / [B,H,W] / [B,1,H,W]")

    # 2. Normalize to {0, 1}
    mask01 = (mask > 0).float()

    # 3. max_pool as dilation
    k   = 2 * n + 1         
    pad = n
    dilated01 = F.max_pool2d(mask01, kernel_size=k, stride=1, padding=pad)

    # 4. Restore to 0/keep_value and the original shape/dtype
    dilated = dilated01 > 0.5
    if keep_value != 1:
        dilated = dilated.to(torch.uint8) * keep_value
    else:
        dilated = dilated.to(torch.uint8)

    if orig_shape == dilated.shape:
        return dilated.to(orig_dtype)
    if len(orig_shape) == 2:          # [H,W]
        return dilated.squeeze(0).squeeze(0).to(orig_dtype)
    elif len(orig_shape) == 3:        # [B,H,W]
        return dilated.squeeze(1).to(orig_dtype)
    else:                             # [B,1,H,W]
        return dilated.to(orig_dtype)

def get_sum_activation_score_batch(
        gt_mask_uint8: torch.Tensor,   # (H, W)
        diff_map     : torch.Tensor,   # (B, H, W)
        eps: float = 1e-8
) -> torch.Tensor:                    # (B,) one score per sample
    """
    Compute the total activation score between each diff_map and a shared gt_mask_uint8.

    gt_mask_uint8 : uint8, foreground == 255
    diff_map      : signed real values, shape (B, H, W)
    """
    # |diff|
    diff = diff_map.abs()             # (B, H, W)

    B = diff.size(0)                  # batch size
    N = diff[0].numel()               # H*W

    # Per-sample softmax over all pixels
    prob = torch.softmax(diff.view(B, -1), dim=1)     # (B, N)

    # Binarize and flatten the GT mask
    G = (gt_mask_uint8 >= 255).float().flatten()      # (N,)
    G = G.to(prob.device)

    # Dot product -> per-sample score
    scores = prob.matmul(G)       # (B,)

    return scores

class criterion_early_stop_strategy():
    def __init__(self, device, model_clip_name="ViT-L/14", model_dino_name="dinov2_vitl14") -> None:
        
        self.device = device 

        # ----- Pixel-preservation metrics ----- #
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        self.ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        self.l1_metric_calculator = nn.L1Loss()
        self.l2_metric_calculator = nn.MSELoss()

        # ----- CLIP ----- #
        if model_clip_name == "ViT-L/14":
            self.model_clip, self.transform_clip = clip.load("ViT-L/14", device=self.device, download_root="../cache_model/clip")

        # ----- DINO ----- #
        if model_dino_name == "dino_vitb16":
            self.model_dino = torch.hub.load("../cache_model/torch/hub/facebookresearch_dino_main", 'dino_vitb16', source="local")
        elif model_dino_name == "dinov2_vitl14":
            self.model_dino = torch.hub.load("../cache_model/torch/hub/dinov2", 'dinov2_vitl14', source="local")
            # self.model_dino = torch.hub.load("../cache_temp/torch/hub/dinov2", 'dinov2_vitl14', source="local")

        self.model_dino.eval()
        self.model_dino.to("cuda")
        self.transform_dino = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def encode(self, image, model, transform, metric, mask=None):
        
        if mask is not None:
            img_np = np.asarray(image)
            mask_np = (mask > 0).astype(np.uint8)[..., None]   
            img_np  = img_np * mask_np
            image = Image.fromarray(img_np)
        
        # Ensure RGB (handles RGBA and other modes)
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')

        image_input = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if metric == 'clip_i':
                image_features = model.encode_image(image_input).detach().float()
            elif metric == 'dino':
                image_features = model(image_input).detach().float()

        return image_features
    
    def encode_batch(self, image_path_list, model, transform, metric, mask=None, image_info=None):

        image_input_all = None 

        for idx_img, img_path in enumerate(image_path_list):
            if idx_img == 0:
                image = Image.open(img_path).convert('RGB')
                if image_info is not None:
                    if image.size != image_info:
                        image = image.resize(image_info)
                if mask is not None:
                    img_np = np.asarray(image)
                    mask_np = (mask > 0).astype(np.uint8)[..., None]   
                    img_np  = img_np * mask_np
                    image = Image.fromarray(img_np)

                image_input_all = transform(image).unsqueeze(0).to(self.device)
            else:
                image = Image.open(img_path).convert('RGB')
                if image_info is not None:
                    if image.size != image_info:
                        image = image.resize(image_info)
                if mask is not None:
                    img_np = np.asarray(image)
                    mask_np = (mask > 0).astype(np.uint8)[..., None]   
                    img_np  = img_np * mask_np
                    image = Image.fromarray(img_np)
                image_input_all = torch.cat((image_input_all, transform(image).unsqueeze(0).to(self.device)))

        with torch.no_grad():
            if metric == 'clip_i':
                image_features_all = model.encode_image(image_input_all).detach().float()
            elif metric == 'dino':
                image_features_all = model(image_input_all).detach().float()
        return image_features_all

    def calculate_metric_l1_or_l2(self, img_pair, metric):

        img1 = transforms.ToTensor()(img_pair[0])
        img2 = transforms.ToTensor()(img_pair[1])

        if metric == "l1":
            score = self.l1_metric_calculator(img1, img2).detach().cpu().numpy().item()
        elif metric == "l2":
            score = self.l2_metric_calculator(img1, img2).detach().cpu().numpy().item()

        return score

    def calculate_metric_model_cos_sim(self, img_pair, metric='clip_i'):
        
        if metric == "clip_i":
            img_feature_1 = self.encode(img_pair[0], self.model_clip, self.transform_clip)
            img_feature_2 = self.encode(img_pair[1], self.model_clip, self.transform_clip)
        elif metric == "dino":
            img_feature_1 = self.encode(img_pair[0], self.model_dino, self.transform_dino)
            img_feature_2 = self.encode(img_pair[1], self.model_dino, self.transform_dino)

        score = 1 - spatial.distance.cosine(img_feature_1.view(img_feature_1.shape[1]),
                                            img_feature_2.view(img_feature_2.shape[1]))
        
        return score 

    def calculate_metric_lpips_or_ssim(self, img_pair, metric="lpips", mask_pair=None):

        img1 = np.array(img_pair[0]).astype(np.float32) / 255
        img2 = np.array(img_pair[1]).astype(np.float32) / 255
        assert img1.shape == img2.shape, "Image shapes should be the same."

        if mask_pair is not None:
            mask1 = np.array(mask_pair[0]).astype(np.float32)
            mask2 = np.array(mask_pair[1]).astype(np.float32)
            img1 = img1 * mask1
            img2 = img2 * mask2

        img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if metric == "lpips":
            score = self.lpips_metric_calculator(img1 * 2 - 1, img2 * 2 - 1)
            score = score.cpu().item()
            return score 
        elif metric == "ssim": 
            score = self.ssim_metric_calculator(img1, img2)
            score = score.cpu().item()
            return score 
        elif metric == "lpips_ssim":
            score_lpips = self.lpips_metric_calculator(img1 * 2 - 1, img2 * 2 - 1)
            score_ssim = self.ssim_metric_calculator(img1, img2)
            return score_lpips.cpu().item(), score_ssim.cpu().item()

    def judge_similar_image_group(self, original_img, img_path_list, threshold=0.95, threshold_diff=0.95, mean_crop_thred=0.95, max_group_size=8, path_mask_image=None, remove_sim_way="dino"):

        data_mask = None
        if path_mask_image is not None:
            data_mask = Image.open(path_mask_image).convert("L")
            data_mask = np.asarray(data_mask)

        image_info = original_img.size 

        # ---------- CLIP similarity ---------- #
        if "clip" in remove_sim_way:
            clip_img_feature_original = self.encode(original_img, self.model_clip, self.transform_clip, metric="clip_i", mask=data_mask)
            clip_img_feature_original = clip_img_feature_original / clip_img_feature_original.norm(dim=-1, keepdim=True)

            clip_img_feature_all = self.encode_batch(img_path_list, self.model_clip, self.transform_clip, metric="clip_i", mask=data_mask, image_info=image_info)
            clip_img_feature_all = clip_img_feature_all / clip_img_feature_all.norm(dim=-1, keepdim=True)

            clip_feature_all = torch.cat((clip_img_feature_original, clip_img_feature_all))

            clip_feature_all = clip_feature_all / clip_feature_all.norm(dim=-1, keepdim=True)

            clip_sim_matric = clip_feature_all @ clip_feature_all.t()

            clip_sim_metric_mean = clip_sim_matric[1:, 1:].mean()

        # ---------- DINO similarity ---------- #
        if "dino" in remove_sim_way:
            dino_img_feature_original = self.encode(original_img, self.model_dino, self.transform_dino, metric="dino", mask=data_mask) 
            dino_img_feature_original = dino_img_feature_original / dino_img_feature_original.norm(dim=-1, keepdim=True)

            dino_img_feature_all = self.encode_batch(img_path_list, self.model_dino, self.transform_dino, metric="dino", mask=data_mask, image_info=image_info)
            dino_img_feature_all = dino_img_feature_all / dino_img_feature_all.norm(dim=-1, keepdim=True)

            dino_feature_all = torch.cat((dino_img_feature_original, dino_img_feature_all))
            # dino_feature_all = dino_img_feature_all - dino_img_feature_original
            # dino_feature_all = dino_img_feature_all

            dino_feature_all = dino_feature_all / dino_feature_all.norm(dim=-1, keepdim=True)

            # dino_sim_original = dino_img_feature_original @ dino_img_feature_all.t()
            dino_sim_metric = dino_feature_all @ dino_feature_all.t()
            # dino_sim_metric = torch.cat((dino_sim_metric, dino_sim_original))

            dino_sim_metric_mean = dino_sim_metric[1:, 1:].mean() 

        # ---------- Grouping ---------- #
        diff_flag = 0

        if remove_sim_way == "dino":
            if dino_sim_metric_mean >= mean_crop_thred:
                diff_flag = 1
        elif remove_sim_way == "clip":
            if clip_sim_metric_mean >= mean_crop_thred:
                diff_flag = 1
        else:
            if clip_sim_metric_mean >= mean_crop_thred and dino_sim_metric_mean >= mean_crop_thred:
                diff_flag = 1 

        if diff_flag:

            if "clip" in remove_sim_way:
                # ----- Compute differential features ----- #
                clip_feature_all_diff = clip_img_feature_all - clip_img_feature_original 
                clip_feature_all_diff = clip_feature_all_diff / clip_feature_all_diff.norm(dim=-1, keepdim=True)
                clip_sim_diff_matric = clip_feature_all_diff @ clip_feature_all_diff.t()

                clip_sim_matric = clip_sim_diff_matric

            if "dino" in remove_sim_way:
                dino_feature_all_diff = dino_img_feature_all - dino_img_feature_original
                dino_feature_all_diff = dino_feature_all_diff / dino_feature_all_diff.norm(dim=-1, keepdim=True)
                dino_sim_diff_metric = dino_feature_all_diff @ dino_feature_all_diff.t()

                dino_sim_metric = dino_sim_diff_metric

            threshold = threshold_diff

        group_dict = {}
        idx_to_group_dict = {}

        record_list = list(range(len(img_path_list)))

        # ----- Remove similar candidates ----- #
        # 1) Build a final similarity matrix.
        if remove_sim_way == "dino":
            sim_mat = dino_sim_metric 
        elif remove_sim_way == "clip":
            sim_mat = clip_sim_matric
        elif remove_sim_way == "clip_and_dino":
            sim_mat = torch.min(clip_sim_matric, dino_sim_metric).clone()
        elif remove_sim_way == "clip_or_dino":
            sim_mat = torch.max(clip_sim_matric, dino_sim_metric).clone()

        if diff_flag == 0:
            sim_mat = sim_mat[1:, 1:]

        sim_mat.fill_diagonal_(-1.)  # ignore self-pairs

        N = sim_mat.size(0)

        # 2) Collect pairs with sim >= threshold, sorted by similarity (descending).
        idx_pairs = torch.nonzero(sim_mat >= threshold)         # (K, 2)
        sims      = sim_mat[idx_pairs[:, 0], idx_pairs[:, 1]]    # (K,)
        order     = torch.argsort(sims, descending=True)
        sorted_pairs = idx_pairs[order]                          # (K, 2)

        # 3) Union-find + per-root member set.
        parent = list(range(N))
        members = {i: set([i]) for i in range(N)}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in sorted_pairs.tolist():

            gi, gj = find(i), find(j)
            if gi == gj:          # already merged
                continue

            # Skip if either group is already too large.
            if len(members[gi]) >= max_group_size or len(members[gj]) >= max_group_size:
                continue 

            # 3.1) Check whether the two groups can be merged:
            # every cross-group pair must satisfy sim >= threshold.
            can_merge = True
            # Iterate over the smaller group for efficiency.
            if len(members[gi]) > len(members[gj]):
                gi, gj = gj, gi

            for u in members[gi]:
                # Vectorised comparison of u vs all members of gj.
                v_idx   = torch.tensor(list(members[gj]), device=sim_mat.device)
                if (sim_mat[u, v_idx] < threshold).any():
                    can_merge = False
                    break

            # 3.2) Actual merge.
            if can_merge:
                parent[gi] = gj
                members[gj].update(members[gi])
                del members[gi]

        # 4) Assemble output.
        group_dict = {}
        idx_to_group = {}
        for g_id, (root, mem_set) in enumerate(members.items()):
            group_dict[g_id] = sorted(mem_set)
            for idx in mem_set:
                idx_to_group[idx] = g_id

        if "delete_no_change" in group_dict:
            del group_dict["delete_no_change"]

        return group_dict, idx_to_group_dict


    def get_clip_and_dino_feature(self, original_img, img_path_list):
        
        clip_img_feature_original = self.encode(original_img, self.model_clip, self.transform_clip, metric="clip_i")
        clip_img_feature_original = clip_img_feature_original / clip_img_feature_original.norm(dim=-1, keepdim=True)
        
        clip_img_feature_all = self.encode_batch(img_path_list, self.model_clip, self.transform_clip, metric="clip_i")
        clip_img_feature_all = clip_img_feature_all / clip_img_feature_all.norm(dim=-1, keepdim=True)

        clip_img_diff_feature = clip_img_feature_all - clip_img_feature_original
        clip_img_diff_feature = clip_img_diff_feature / clip_img_diff_feature.norm(dim=-1, keepdim=True)

        dino_img_feature_original = self.encode(original_img, self.model_dino, self.transform_dino, metric="dino")
        dino_img_feature_original = dino_img_feature_original / dino_img_feature_original.norm(dim=-1, keepdim=True)

        dino_img_feature_all = self.encode_batch(img_path_list, self.model_dino, self.transform_dino, metric="dino")
        dino_img_feature_all = dino_img_feature_all / dino_img_feature_all.norm(dim=-1, keepdim=True)


        dino_img_diff_feature = dino_img_feature_all - dino_img_feature_original
        dino_img_diff_feature = dino_img_diff_feature / dino_img_diff_feature.norm(dim=-1, keepdim=True)

        return clip_img_diff_feature, dino_img_diff_feature
    
    def get_score_by_input_and_output_caption(self, original_img=None, img_path_list=None, input_caption=None, output_caption=None, num_task_key=None, caption_retain_rate=0.8, sim_ti_list=None, cos_direction_list=None, return_score_flag=False):
        """Score candidates using input and output captions."""

        select_num = int(num_task_key * caption_retain_rate)
        if len(img_path_list) < select_num:
            return list(range(len(img_path_list))), list(range(len(img_path_list)))
        
        if sim_ti_list is None or cos_direction_list is None:

            # ---------- CLIP similarity ---------- #
            clip_img_feature_original = self.encode(original_img, self.model_clip, self.transform_clip, metric="clip_i")
            clip_img_feature_original = clip_img_feature_original / clip_img_feature_original.norm(dim=-1, keepdim=True)

            clip_img_feature_all = self.encode_batch(img_path_list, self.model_clip, self.transform_clip, metric="clip_i")
            clip_img_feature_all = clip_img_feature_all / clip_img_feature_all.norm(dim=-1, keepdim=True)

            # ---------- DINO similarity ---------- #
            dino_img_feature_original = self.encode(original_img, self.model_dino, self.transform_dino, metric="dino")
            dino_img_feature_original = dino_img_feature_original / dino_img_feature_original.norm(dim=-1, keepdim=True)

            dino_img_feature_all = self.encode_batch(img_path_list, self.model_dino, self.transform_dino, metric="dino")
            dino_img_feature_all = dino_img_feature_all / dino_img_feature_all.norm(dim=-1, keepdim=True)

            # ---------- Text similarity ---------- #
            input_text_tokens = clip.tokenize(input_caption, truncate=True).to("cuda")
            input_text_feature = self.model_clip.encode_text(input_text_tokens).detach().float()
            input_text_feature = input_text_feature / input_text_feature.norm(dim=-1, keepdim=True)

            output_text_tokens = clip.tokenize(output_caption, truncate=True).to("cuda")
            output_text_feature = self.model_clip.encode_text(output_text_tokens).detach().float()
            output_text_feature = output_text_feature / output_text_feature.norm(dim=-1, keepdim=True)

            # ----- output caption vs edited image similarity ----- #
            sim_ti_input = input_text_feature @ clip_img_feature_original.t()

            sim_ti_output = output_text_feature @ clip_img_feature_all.t()


            # ----- Directional consistency ----- #
            eps = 1e-8 
            delta_T = output_text_feature - input_text_feature
            delta_T_norm = delta_T / (delta_T.norm(dim=-1, keepdim=True) + eps)

            delta_I_clip = clip_img_feature_all - clip_img_feature_original
            delta_I_clip_norm = delta_I_clip / (delta_I_clip.norm(dim=-1, keepdim=True) + eps)

            cos_direction_clip = (delta_I_clip_norm @ delta_T_norm.t()).squeeze(-1)       

            if return_score_flag:
                return sim_ti_output[0], cos_direction_clip

            sim_ti_top, indices_sim_ti = torch.topk(sim_ti_output[0], k=select_num)
            cos_direction_clip_top, indices_cos_direction = torch.topk(cos_direction_clip, k=select_num)

            indices_sim_ti_list = indices_sim_ti.cpu().tolist()
            indices_cos_direction_list = indices_cos_direction.cpu().tolist()

        else:
            
            sim_ti_output = torch.tensor(sim_ti_list).to(self.device)
            cos_direction_clip = torch.tensor(cos_direction_list).to(self.device)

            sim_ti_top, indices_sim_ti = torch.topk(sim_ti_output, k=select_num)
            cos_direction_clip_top, indices_cos_direction = torch.topk(cos_direction_clip, k=select_num)

            indices_sim_ti_list = indices_sim_ti.cpu().tolist()
            indices_cos_direction_list = indices_cos_direction.cpu().tolist()


        return indices_sim_ti_list, indices_cos_direction_list
    
    def get_score_by_output_caption(self, original_img=None, img_path_list=None, output_caption=None, num_task_key=None, caption_retain_rate=0.8, return_score_flag=False):
        """Score candidates using the output caption only."""

        select_num = int(num_task_key * caption_retain_rate)
        if len(img_path_list) < select_num:
            return list(range(len(img_path_list))), list(range(len(img_path_list)))
        

        # ---------- CLIP similarity ---------- #
        clip_img_feature_all = self.encode_batch(img_path_list, self.model_clip, self.transform_clip, metric="clip_i")
        clip_img_feature_all = clip_img_feature_all / clip_img_feature_all.norm(dim=-1, keepdim=True)

        # ---------- Text similarity ---------- #
        output_text_tokens = clip.tokenize(output_caption, truncate=True).to("cuda")
        output_text_feature = self.model_clip.encode_text(output_text_tokens).detach().float()
        output_text_feature = output_text_feature / output_text_feature.norm(dim=-1, keepdim=True)

        # ----- output caption vs edited image similarity ----- #
        sim_ti_output = output_text_feature @ clip_img_feature_all.t()

        if return_score_flag:
            return sim_ti_output[0]

        sim_ti_top, indices_sim_ti = torch.topk(sim_ti_output[0], k=select_num)

        indices_sim_ti_list = indices_sim_ti.cpu().tolist()

        return indices_sim_ti_list

    def get_select_idx_by_edited_region(self, original_img, img_path_list, path_mask_image, num_task_key, kernel=(16, 16), stride=(16, 16), padding_num=2, delete_region_retain_rate=0.5):
        """Score candidates from the edited region."""

        select_num = int(num_task_key * delete_region_retain_rate)
        if len(img_path_list) < select_num:
            return list(range(len(img_path_list)))

        input_image_modified, img_info = input_process_image(original_img)
        input_image_tensor = load_image(input_image_modified).to(self.device)

        data_mask = Image.open(path_mask_image).convert("L")
        data_mask = data_mask.resize(input_image_modified.size)
        data_mask = np.array(data_mask)
        data_mask_tensor = torch.from_numpy(data_mask).to(self.device)
        data_mask_tensor_slide =  sliding_window_sum(data_mask_tensor, kernel=kernel, stride=stride)

        # data_mask_aggregation = (data_mask_tensor_slide >=255).cpu().numpy().astype(np.uint8) * 255

        diff_image_all = None
        for output_idx, path_output_image in enumerate(img_path_list):
            output_image = Image.open(path_output_image).convert('RGB')

            output_image = output_image.resize(input_image_modified.size)
            output_image_tensor = load_image(output_image).to(self.device)

            diff_image = output_image_tensor - input_image_tensor
            diff_image = diff_image[0].mean(dim=0)

            diff_image = sliding_window_sum(diff_image, kernel=kernel, stride=stride) # kernel=(8,8), stride=(2,2)

            if diff_image_all is None:
                diff_image_all = diff_image.unsqueeze(0)
            else:
                diff_image_all = torch.cat((diff_image_all, diff_image.unsqueeze(0)))


        sum_activation_score = get_sum_activation_score_batch(data_mask_tensor_slide, diff_image_all)
        num_success = (sum_activation_score > 0.98).sum().item()

        data_mask_tensor_slide = dilate_mask(data_mask_tensor_slide, n=padding_num) 
        
        # if num_success < int(num_task_key*0.8):
        #     data_mask_tensor_slide = dilate_mask(data_mask_tensor_slide, n=padding_num) 
        #     sum_activation_score = get_sum_activation_score_batch(data_mask_tensor_slide, diff_image_all)
        #     num_success = (sum_activation_score > 0.98).sum().item()
        
        while num_success < select_num:
            data_mask_tensor_slide = dilate_mask(data_mask_tensor_slide, n=padding_num) 
            sum_activation_score = get_sum_activation_score_batch(data_mask_tensor_slide, diff_image_all)
            num_success = (sum_activation_score > 0.98).sum().item()

        select_idx_list = (sum_activation_score > 0.98).nonzero(as_tuple=True)[0].cpu().tolist()

        if len(select_idx_list) < num_task_key:
            print(f"\nSaved samples: ", num_task_key - len(select_idx_list))

        return select_idx_list


    def get_score_by_edited_region(self, original_img, img_path_list, path_mask_image, kernel=(16, 16), stride=(16, 16), padding_num=2):
        """Score candidates from the edited region."""

        input_image_modified, img_info = input_process_image(original_img)
        input_image_tensor = load_image(input_image_modified).to(self.device)

        data_mask = Image.open(path_mask_image).convert("L")
        data_mask = data_mask.resize(input_image_modified.size)
        data_mask = np.array(data_mask)
        data_mask_tensor = torch.from_numpy(data_mask).to(self.device)
        data_mask_tensor_slide =  sliding_window_sum(data_mask_tensor, kernel=kernel, stride=stride)

        diff_image_all = None
        for output_idx, path_output_image in enumerate(img_path_list):
            output_image = Image.open(path_output_image).convert('RGB')

            output_image = output_image.resize(input_image_modified.size)
            output_image_tensor = load_image(output_image).to(self.device)

            diff_image = output_image_tensor - input_image_tensor
            diff_image = diff_image[0].mean(dim=0)

            diff_image = sliding_window_sum(diff_image, kernel=kernel, stride=stride) # kernel=(8,8), stride=(2,2)

            if diff_image_all is None:
                diff_image_all = diff_image.unsqueeze(0)
            else:
                diff_image_all = torch.cat((diff_image_all, diff_image.unsqueeze(0)))
        
        sum_activation_score = get_sum_activation_score_batch(data_mask_tensor_slide, diff_image_all) 

        score_list = sum_activation_score.detach().cpu().tolist()  

        return score_list

    def get_score_by_edited_region_by_padding_1025(self, original_img, img_path_list, path_mask_image, kernel=(16, 16), stride=(16, 16), padding_num=2):
        """Score candidates from the edited region (with padding expansion)."""

        input_image_modified, img_info = input_process_image(original_img)
        input_image_tensor = load_image(input_image_modified).to(self.device)

        data_mask = Image.open(path_mask_image).convert("L")
        data_mask = data_mask.resize(input_image_modified.size)
        data_mask = np.array(data_mask)
        data_mask_tensor = torch.from_numpy(data_mask).to(self.device)
        data_mask_tensor_slide =  sliding_window_sum(data_mask_tensor, kernel=kernel, stride=stride)

        diff_image_all = None
        for output_idx, path_output_image in enumerate(img_path_list):
            output_image = Image.open(path_output_image).convert('RGB')

            output_image = output_image.resize(input_image_modified.size)
            output_image_tensor = load_image(output_image).to(self.device)

            diff_image = output_image_tensor - input_image_tensor
            diff_image = diff_image[0].mean(dim=0)

            diff_image = sliding_window_sum(diff_image, kernel=kernel, stride=stride) # kernel=(8,8), stride=(2,2)

            if diff_image_all is None:
                diff_image_all = diff_image.unsqueeze(0)
            else:
                diff_image_all = torch.cat((diff_image_all, diff_image.unsqueeze(0)))

        img_to_score_dict = {}

        idx_pad = 0

        num_success = 0
        while num_success < len(img_path_list):
            sum_activation_score = get_sum_activation_score_batch(data_mask_tensor_slide, diff_image_all)
            num_success = (sum_activation_score >= 0.98).sum().item()

            # Assign scores
            success_idx_list = (sum_activation_score >= 0.98).nonzero(as_tuple=True)[0].cpu().tolist()

            for temp_idx in success_idx_list:
                if img_path_list[temp_idx] not in img_to_score_dict:
                    img_to_score_dict[img_path_list[temp_idx]] = idx_pad

            data_mask_tensor_slide = dilate_mask(data_mask_tensor_slide, n=padding_num) 

            idx_pad += 1

        # print("num_success: ", num_success)
        # print("len(img_path_list): ", len(img_path_list))
        # print("img_to_score_dict: ", img_to_score_dict)

        score_list = []
        for temp_img_path in img_path_list:
            score_list.append(img_to_score_dict[temp_img_path])

        return score_list