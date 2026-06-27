# ==================== Import Packages ==================== #
import time
import sys
import os 

from pathlib import Path
import datetime

import functools

import numpy as np
import json 

import math
import sampling

import torch
from einops import rearrange, repeat
from PIL import Image, ImageOps
from safetensors.torch import load_file
from torchvision.transforms import functional as F

from torch import Tensor
import torch.distributed as dist
from xfuser.core.distributed import (
    get_world_group,
    initialize_model_parallel,
)

from tqdm import tqdm 
import itertools

# ----- Step1X-Edit imports ----- #
from modules.autoencoder import AutoEncoder
from modules.conditioner import Qwen25VL_7b_Embedder as Qwen2VLEmbedder
from modules.model_edit import Step1XParams, Step1XEdit
from modules.multigpu import parallel_transformer, teacache_transformer, parallel_teacache_transformer


# ==================== Constant Parameters ==================== #


# ==================== Functions ==================== #

def cudagc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def cfg_usp_level_setting(ring_degree: int = 1, ulysses_degree: int = 1, cfg_degree: int = 1):
    # restriction: dist.get_world_size() == <cfg_degree> x <ring_degree> x <ulysses_degree>
    initialize_model_parallel(
        ring_degree=ring_degree,
        ulysses_degree=ulysses_degree,
        classifier_free_guidance_degree=cfg_degree,
    )

def teacache_init(pipe, args):
    pipe.dit.__class__.enable_teacache = True
    pipe.dit.__class__.cnt = 0
    pipe.dit.__class__.num_steps = args.num_steps
    pipe.dit.__class__.rel_l1_thresh = args.teacache_threshold
    pipe.dit.__class__.accumulated_rel_l1_distance = 0
    pipe.dit.__class__.previous_modulated_input = None
    pipe.dit.__class__.previous_residual = None
    
    print(f"[teacache_init] teacache parameters:")
    print(f"  - num_steps: {args.num_steps}")
    print(f"  - teacache_threshold: {args.teacache_threshold}")
    print(f"  - enable_teacache: {pipe.dit.__class__.enable_teacache}")


def load_state_dict(model, ckpt_path, device="cuda", strict=False, assign=True):
    if Path(ckpt_path).suffix == ".safetensors":
        state_dict = load_file(ckpt_path, device)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(
        state_dict, strict=strict, assign=assign
    )
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    return model


def load_models(
    dit_path=None,
    ae_path=None,
    qwen2vl_model_path=None,
    mode="flash",
    device="cuda",
    max_length=256,
    dtype=torch.bfloat16,
    prefix_way="default"
):
    qwen2vl_encoder = Qwen2VLEmbedder(
        qwen2vl_model_path,
        device=device,
        max_length=max_length,
        dtype=dtype,
        prefix_way=prefix_way
    )

    with torch.device("meta"):
        ae = AutoEncoder(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        )

        step1x_params = Step1XParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            mode=mode
        )
        dit = Step1XEdit(step1x_params)

    ae = load_state_dict(ae, ae_path, 'cpu')
    dit = load_state_dict(
        dit, dit_path, 'cpu'
    )

    ae = ae.to(dtype=torch.float32)

    return ae, dit, qwen2vl_encoder

def equip_dit_with_lora_sd_scripts(ae, text_encoders, dit, lora, device='cuda'):
    from safetensors.torch import load_file
    weights_sd = load_file(lora)
    is_lora = True
    from library import lora_module
    module = lora_module
    lora_model, _ = module.create_network_from_weights(1.0, None, ae, text_encoders, dit, weights_sd, True)
    lora_model.merge_to(text_encoders, dit, weights_sd)

    lora_model.set_multiplier(1.0)
    return lora_model

class ImageGenerator:
    def __init__(
        self,
        dit_path=None,
        ae_path=None,
        qwen2vl_model_path=None,
        device="cuda",
        max_length=640,
        dtype=torch.bfloat16,
        quantized=False,
        offload=False,
        lora=None,
        mode="flash", 
        prefix_way="default",
        
    ) -> None:
        if os.getenv("TORCHELASTIC_RUN_ID") is not None:
            local_rank = get_world_group().local_rank
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device(device)

        self.ae, self.dit, self.llm_encoder = load_models(
            dit_path=dit_path,
            ae_path=ae_path,
            qwen2vl_model_path=qwen2vl_model_path,
            max_length=max_length,
            dtype=dtype,
            device=self.device,
            mode=mode, 
            prefix_way=prefix_way,
        )
        if not quantized:
            self.dit = self.dit.to(dtype=torch.bfloat16)
        else:
            self.dit = self.dit.to(dtype=torch.float8_e4m3fn)
        if not offload:
            self.dit = self.dit.to(device=self.device)
            self.ae = self.ae.to(device=self.device)
        self.quantized = quantized 
        self.offload = offload
        if lora is not None:
            self.lora_module = equip_dit_with_lora_sd_scripts(
                self.ae,
                [self.llm_encoder],
                self.dit,
                lora,
                device=self.dit.device,
            )
        else:
            self.lora_module = None
        self.mode = mode

        # ----- Added fields ----- #
        self.path_save_pt_output = None
        self.path_save_output_xt_to_x0 = None

        self.width = None 
        self.height = None 

        self.img_info = None 

        self.path_resume_pt_output = None  


    def prepare(self, prompt, img, ref_image, ref_image_raw):
        bs, _, h, w = img.shape
        bs, _, ref_h, ref_w = ref_image.shape

        assert h == ref_h and w == ref_w

        if bs == 1 and not isinstance(prompt, str):
            bs = len(prompt)
        elif bs >= 1 and isinstance(prompt, str):
            prompt = [prompt] * bs

        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        ref_img = rearrange(ref_image, "b c (ref_h ph) (ref_w pw) -> b (ref_h ref_w) (c ph pw)", ph=2, pw=2)
        if img.shape[0] == 1 and bs > 1:
            img = repeat(img, "1 ... -> bs ...", bs=bs)
            ref_img = repeat(ref_img, "1 ... -> bs ...", bs=bs)

        img_ids = torch.zeros(h // 2, w // 2, 3)

        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        ref_img_ids = torch.zeros(ref_h // 2, ref_w // 2, 3)

        ref_img_ids[..., 1] = ref_img_ids[..., 1] + torch.arange(ref_h // 2)[:, None]
        ref_img_ids[..., 2] = ref_img_ids[..., 2] + torch.arange(ref_w // 2)[None, :]
        ref_img_ids = repeat(ref_img_ids, "ref_h ref_w c -> b (ref_h ref_w) c", b=bs)

        if isinstance(prompt, str):
            prompt = [prompt]
        if self.offload:
            self.llm_encoder = self.llm_encoder.to(self.device)
        
        llm_encoder_output_dict = self.llm_encoder(prompt, ref_image_raw)
        txt = llm_encoder_output_dict["txt"]
        mask = llm_encoder_output_dict["mask"]


        if self.offload:
            self.llm_encoder = self.llm_encoder.cpu()
            cudagc()

        txt_ids = torch.zeros(bs, txt.shape[1], 3)

        img = torch.cat([img, ref_img.to(device=img.device, dtype=img.dtype)], dim=-2)
        img_ids = torch.cat([img_ids, ref_img_ids], dim=-2)


        return {
            "img": img,
            "mask": mask,
            "img_ids": img_ids.to(img.device),
            "llm_embedding": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
        }

    @staticmethod
    def process_diff_norm(diff_norm, k):
        pow_result = torch.pow(diff_norm, k)

        result = torch.where(
            diff_norm > 1.0,
            pow_result,
            torch.where(diff_norm < 1.0, torch.ones_like(diff_norm), diff_norm),
        )
        return result
    
    def get_x_to_vae(self, x, title): 
        """Decode x_t through the VAE into an image."""

        x = self.unpack(x.float(), self.height, self.width)
        if self.offload:
            self.ae = self.ae.to(self.device)
        x = self.ae.decode(x)
        if self.offload:
            self.ae = self.ae.cpu()
            cudagc()
        x = x.clamp(-1, 1)
        x = x.mul(0.5).add(0.5)

        output_image = self.output_process_image(F.to_pil_image(x[0].float()), self.img_info)
        output_image.save(os.path.join(self.path_save_output_xt_to_x0, f"{title}.png"))

        return output_image

    def denoise_one_step_from_t_prev_to_t_curr(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        llm_embedding: torch.Tensor,
        txt_ids: torch.Tensor,
        t_curr,
        t_prev,
        cfg_guidance: float = 4.5,
        mask=None,
        show_progress=False,
        timesteps_truncate=1.0,
    ):
        if self.offload:
            self.dit = self.dit.to(self.device)
        
        # ----- Denoise one step from t_curr to t_prev ----- #
        if img.shape[0] == 1 and cfg_guidance != -1:
            img = torch.cat([img, img], dim=0)
        t_vec = torch.full(
            (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
        )

        pred = self.dit(
            img=img,
            img_ids=img_ids,
            txt_ids=txt_ids,
            timesteps=t_vec,
            llm_embedding=llm_embedding,
            t_vec=t_vec,
            mask=mask,
        )

        if cfg_guidance != -1:
            cond, uncond = (
                pred[0 : pred.shape[0] // 2, :],
                pred[pred.shape[0] // 2 :, :],
            )
            if t_curr > timesteps_truncate:
                diff = cond - uncond
                diff_norm = torch.norm(diff, dim=(2), keepdim=True)
                pred = uncond + cfg_guidance * (
                    cond - uncond
                ) / self.process_diff_norm(diff_norm, k=0.4)
            else:
                pred = uncond + cfg_guidance * (cond - uncond)
        tem_img = img[0 : img.shape[0] // 2, :] + (t_prev - t_curr) * pred
        img_input_length = img.shape[1] // 2
        img = torch.cat(
            [
            tem_img[:, :img_input_length],
            img[ : img.shape[0] // 2, img_input_length:],
            ], dim=1
        )

        if self.offload:
            self.dit = self.dit.cpu()
            cudagc()

        return img

    def denoise_original(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        llm_embedding: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: list[float],
        cfg_guidance: float = 4.5,
        mask=None,
        show_progress=False,
        timesteps_truncate=1.0,
    ):
        if self.offload:
            self.dit = self.dit.to(self.device)
        if show_progress:
            pbar = tqdm(itertools.pairwise(timesteps), desc='denoising...')
        else:
            pbar = itertools.pairwise(timesteps)

        # print("----- before -----") 

        # print("1 - img.shape: ", img.shape) 

        for t_curr, t_prev in pbar:

            if img.shape[0] == 1 and cfg_guidance != -1:
                img = torch.cat([img, img], dim=0)

            # print("2 - img.shape: ", img.shape) 

            t_vec = torch.full(
                (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
            )

            pred = self.dit(
                img=img,
                img_ids=img_ids,
                txt_ids=txt_ids,
                timesteps=t_vec,
                llm_embedding=llm_embedding,
                t_vec=t_vec,
                mask=mask,
            )

            if cfg_guidance != -1:
                cond, uncond = (
                    pred[0 : pred.shape[0] // 2, :],
                    pred[pred.shape[0] // 2 :, :],
                )
                if t_curr > timesteps_truncate:
                    diff = cond - uncond
                    diff_norm = torch.norm(diff, dim=(2), keepdim=True)
                    pred = uncond + cfg_guidance * (
                        cond - uncond
                    ) / self.process_diff_norm(diff_norm, k=0.4)
                else:
                    pred = uncond + cfg_guidance * (cond - uncond)

            # print("middle - img: ", img.shape) 
            # print("pred: ", pred.shape) 

            tem_img = img[0 : img.shape[0] // 2, :] + (t_prev - t_curr) * pred

            img_input_length = img.shape[1] // 2

            img = torch.cat(
                [
                tem_img[:, :img_input_length],
                img[ : img.shape[0] // 2, img_input_length:],
                ], dim=1
            )

            # print("3 - img.shape: ", img.shape) 

        if self.offload:
            self.dit = self.dit.cpu()
            cudagc()

        # print("\nt_curr: ", t_curr) 
        # print("t_prev: ", t_prev)

        # print("4 - img.shape: ", img.shape) 

        # return img[:, :img.shape[1] // 2]

        return img, pred



    def denoise(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        llm_embedding: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: list[float],
        cfg_guidance: float = 4.5,
        mask=None,
        show_progress=False,
        timesteps_truncate=1.0,
        way="compute_directly" # "denoise_new_step" # 
    ):
        if self.offload:
            self.dit = self.dit.to(self.device)
        if show_progress:
            pbar = tqdm(itertools.pairwise(timesteps), desc='denoising...')
        else:
            pbar = itertools.pairwise(timesteps)

        idx_step = 0
        for t_curr, t_prev in pbar:

            # print("t_curr: ", t_curr)
            # print("t_prev: ", t_prev)
            # print()

            idx_step += 1

            if img.shape[0] == 1 and cfg_guidance != -1:
                img = torch.cat([img, img], dim=0)
            t_vec = torch.full(
                (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
            )

            pred = self.dit(
                img=img,
                img_ids=img_ids,
                txt_ids=txt_ids,
                timesteps=t_vec,
                llm_embedding=llm_embedding,
                t_vec=t_vec,
                mask=mask,
            )

            if cfg_guidance != -1:
                cond, uncond = (
                    pred[0 : pred.shape[0] // 2, :],
                    pred[pred.shape[0] // 2 :, :],
                )
                if t_curr > timesteps_truncate:
                    diff = cond - uncond
                    diff_norm = torch.norm(diff, dim=(2), keepdim=True)
                    pred = uncond + cfg_guidance * (
                        cond - uncond
                    ) / self.process_diff_norm(diff_norm, k=0.4)
                else:
                    pred = uncond + cfg_guidance * (cond - uncond)
            tem_img = img[0 : img.shape[0] // 2, :] + (t_prev - t_curr) * pred

            img_input_length = img.shape[1] // 2
            

            if self.path_save_pt_output is not None:
                if idx_step in [2, 3, 4, 5, 7,8, 11, 12, 15,16, 19, 20, 23, 24]:
                    if way == "compute_directly":
                        img_x0 = img[0 : img.shape[0] // 2, :] + (0 - t_curr) * pred 
                        # img_x0 = img[0 : img.shape[0] // 2, :]

            img = torch.cat(
                [
                tem_img[:, :img_input_length],
                img[ : img.shape[0] // 2, img_input_length:],
                ], dim=1
            )

            if self.path_save_pt_output is not None:

                if idx_step in [2, 3, 4, 5, 7,8, 11, 12, 15,16, 19, 20, 23, 24]: # 1, 2, 3,  8, 12, 16, 20
                    # Temporarily disable teacache to avoid state-inconsistency tensor-shape mismatches.
                    original_enable = getattr(self.dit, 'enable_teacache', False)
                    self.dit.enable_teacache = False
                    
                    try:
                        torch.save(img.detach().cpu(), os.path.join(self.path_save_pt_output, f"{idx_step}.pt"))

                        # Save img_ids, llm_embedding, txt_ids
                        if not os.path.exists(os.path.join(self.path_save_pt_output, "img_ids.pt")):
                            torch.save(img_ids.detach().cpu(), os.path.join(self.path_save_pt_output, "img_ids.pt"))
                        if not os.path.exists(os.path.join(self.path_save_pt_output, "llm_embedding.pt")):
                            torch.save(llm_embedding.detach().cpu(), os.path.join(self.path_save_pt_output, "llm_embedding.pt"))
                        if not os.path.exists(os.path.join(self.path_save_pt_output, "txt_ids.pt")):
                            torch.save(txt_ids.detach().cpu(), os.path.join(self.path_save_pt_output, "txt_ids.pt"))

                        # self.get_x_to_vae(img[:, :img.shape[1] // 2], title=f"x_t-to_vae-{idx_step}")

                        if way == "denoise_new_step":
                            img_x0 = self.denoise_one_step_from_t_prev_to_t_curr(
                                img=img,
                                img_ids=img_ids,
                                llm_embedding=llm_embedding,
                                txt_ids=txt_ids,
                                t_curr=t_prev,
                                t_prev=0,
                                cfg_guidance=cfg_guidance,
                                mask=mask,
                                show_progress=show_progress,
                                timesteps_truncate=timesteps_truncate,
                            )

                        # print("img_x0: ", img_x0.shape) # [1, 8120, 64]
                        # print("img: ", img.shape) # 

                        # print("img_x0[:, :img.shape[1] // 2]: ", img_x0[:, :img.shape[1] // 2].shape) # [1, 4060, 64]

                        self.get_x_to_vae(img_x0[:, :img.shape[1] // 2], title=f"x_t-x_0-to_vae-{idx_step}")
                    finally:
                        # Restore the original teacache setting even when an exception is raised.
                        self.dit.enable_teacache = original_enable


        if self.offload:
            self.dit = self.dit.cpu()
            cudagc()

        return img[:, :img.shape[1] // 2]

    @staticmethod
    def unpack(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(height / 16),
            w=math.ceil(width / 16),
            ph=2,
            pw=2,
        )

    @staticmethod
    def load_image(image):
        from PIL import Image

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

    def output_process_image(self, resize_img, image_size):
        res_image = resize_img.resize(image_size)
        return res_image
    
    def input_process_image(self, img, img_size=512):
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

    @torch.inference_mode()
    def generate_image(
        self,
        prompt,
        negative_prompt,
        ref_images,
        num_steps,
        cfg_guidance,
        seed,
        num_samples=1,
        init_image=None,
        image2image_strength=0.0,
        show_progress=False,
        size_level=512,
        path_save_pt_output=None,
        path_save_output_xt_to_x0=None,

        # Resume generation from a saved intermediate denoise step.
        path_resume_pt_output=None,  
        num_step_resume=None, 
    ):
        assert num_samples == 1, "num_samples > 1 is not supported yet."
        ref_images_raw, img_info = self.input_process_image(ref_images, img_size=size_level)
        
        width, height = ref_images_raw.width, ref_images_raw.height

        self.path_resume_pt_output = path_resume_pt_output

        self.width = width 
        self.height = height 
        self.img_info = img_info
        self.path_save_pt_output = path_save_pt_output
        self.path_save_output_xt_to_x0 = path_save_output_xt_to_x0

        t0 = time.perf_counter()

        if self.path_resume_pt_output is not None:
            print("resume test !")

            x = torch.load(os.path.join(self.path_resume_pt_output, f"{num_step_resume}.pt")).to(self.device)

            img_ids = torch.load(os.path.join(self.path_resume_pt_output, f"img_ids.pt")).to(self.device)
            llm_embedding = torch.load(os.path.join(self.path_resume_pt_output, f"llm_embedding.pt")).to(self.device)
            txt_ids = torch.load(os.path.join(self.path_resume_pt_output, f"txt_ids.pt")).to(self.device)

            timesteps = sampling.get_schedule(
                num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True
            )

            timesteps = timesteps[num_step_resume:]

            # print("timesteps:", timesteps)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                x = self.denoise(
                    img=x, 
                    img_ids=img_ids,
                    llm_embedding=llm_embedding,
                    txt_ids=txt_ids,
                    cfg_guidance=cfg_guidance,
                    timesteps=timesteps,
                    show_progress=show_progress,
                    timesteps_truncate=1.0,
                )

        else:


            ref_images_raw = self.load_image(ref_images_raw)
            ref_images_raw = ref_images_raw.to(self.device)
            if self.offload:
                self.ae = self.ae.to(self.device)
            ref_images = self.ae.encode(ref_images_raw.to(self.device) * 2 - 1)

            if self.path_save_pt_output is not None:
                torch.save(ref_images.detach().cpu(), os.path.join(self.path_save_pt_output, f"original_x.pt"))

            if self.offload:
                self.ae = self.ae.cpu()
                cudagc()

            seed = int(seed)
            seed = torch.Generator(device="cpu").seed() if seed < 0 else seed

            if init_image is not None:
                init_image = self.load_image(init_image)
                init_image = init_image.to(self.device)
                init_image = torch.nn.functional.interpolate(init_image, (height, width))
                if self.offload:
                    self.ae = self.ae.to(self.device)
                init_image = self.ae.encode(init_image.to() * 2 - 1)
                if self.offload:
                    self.ae = self.ae.cpu()
                    cudagc()
            
            x = torch.randn(
                num_samples,
                16,
                height // 8,
                width // 8,
                device=self.device,
                dtype=torch.bfloat16,
                generator=torch.Generator(device=self.device).manual_seed(seed),
            )

            timesteps = sampling.get_schedule(
                num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True
            )

            if init_image is not None:
                t_idx = int((1 - image2image_strength) * num_steps)
                t = timesteps[t_idx]
                timesteps = timesteps[t_idx:]
                x = t * x + (1.0 - t) * init_image.to(x.dtype)

            x = torch.cat([x, x], dim=0)


            ref_images = torch.cat([ref_images, ref_images], dim=0)
            ref_images_raw = torch.cat([ref_images_raw, ref_images_raw], dim=0)
            inputs = self.prepare([prompt, negative_prompt], x, ref_image=ref_images, ref_image_raw=ref_images_raw)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                x = self.denoise(
                    **inputs,
                    cfg_guidance=cfg_guidance,
                    timesteps=timesteps,
                    show_progress=show_progress,
                    timesteps_truncate=1.0,
                )

        # print(x.shape)
        x = self.unpack(x.float(), height, width)
        if self.offload:
            self.ae = self.ae.to(self.device)
        x = self.ae.decode(x)
        if self.offload:
            self.ae = self.ae.cpu()
            cudagc()
        x = x.clamp(-1, 1)
        x = x.mul(0.5).add(0.5)

        t1 = time.perf_counter()
        if os.getenv("TORCHELASTIC_RUN_ID") is None or dist.get_rank() == 0:
            print(f"Done in {t1 - t0:.1f}s.")
        images_list = []
        for img in x.float():
            images_list.append(self.output_process_image(F.to_pil_image(img), img_info))

        return images_list


    @torch.inference_mode() 
    def generate_final_stage_image(
        self, 
        img, 
        img_ids, 
        llm_embedding, 
        txt_ids, 
        timesteps, 
        cfg_guidance,
        mask, 
        show_progress=True
    ):

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x, pred = self.denoise_original(
                    img=img, 
                    img_ids=img_ids, 
                    llm_embedding=llm_embedding, 
                    txt_ids=txt_ids, 
                    timesteps=timesteps, 
                    cfg_guidance=cfg_guidance,
                    mask=mask, 
                    show_progress=show_progress,
                    timesteps_truncate=1.0,
            )

            # ----- Save early-preview image ----- #
            x = x[:, :x.shape[1] // 2]
            x = self.unpack(x.float(), self.height, self.width)
            if self.offload:
                self.ae = self.ae.to(self.device)
            x = self.ae.decode(x)
            if self.offload:
                self.ae = self.ae.cpu()
                cudagc()
            x = x.clamp(-1, 1)
            x = x.mul(0.5).add(0.5)

            output_image = self.output_process_image(F.to_pil_image(x[0].float()), self.img_info)

        return output_image 

    @torch.inference_mode() 
    def generate_late_stage_image(
        self, 
        img, 
        img_ids, 
        llm_embedding, 
        txt_ids, 
        timesteps, 
        cfg_guidance,
        mask, 
        num_early_steps,
        num_late_steps,
        seed,
        show_progress=True,
        
    ):

        early_timesteps = timesteps[:num_late_steps-num_early_steps+1]
        last_timesteps = timesteps[num_late_steps-num_early_steps+1:]

        # print()
        # print(len(timesteps)) 
        # print(len(early_timesteps))
        # print(len(last_timesteps))
    

        # print("early_timesteps: ", early_timesteps)
        # print("last_timesteps: ", last_timesteps)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x, pred = self.denoise_original(
                    img=img, 
                    img_ids=img_ids, 
                    llm_embedding=llm_embedding, 
                    txt_ids=txt_ids, 
                    timesteps=early_timesteps, 
                    cfg_guidance=cfg_guidance,
                    mask=mask, 
                    show_progress=show_progress,
                    timesteps_truncate=1.0,
            )

            img_x0 = x + (0 - early_timesteps[-1]) * pred  

            # ----- Save early-preview image ----- #
            self.get_x_to_vae(img_x0[:, :x.shape[1] // 2], title=f"{seed}-x_t-x_0-to_vae-{num_late_steps}") 

        return {
            "img": x, 
            "timesteps": last_timesteps,  
        }

    @torch.inference_mode() 
    def generate_early_stage_image(
        self,
        prompt,
        negative_prompt,
        ref_images,
        num_steps,
        cfg_guidance,
        seed,
        num_samples=1,
        init_image=None,
        image2image_strength=0.0,
        show_progress=False,
        size_level=512,

        # ----- Early-stage image output ----- #
        path_save_output_xt_to_x0=None,
        num_early_steps=4 
    ):
        assert num_samples == 1, "num_samples > 1 is not supported yet."

        ref_images_raw, img_info = self.input_process_image(ref_images, img_size=size_level) 

        width, height = ref_images_raw.width, ref_images_raw.height 

        self.width = width 
        self.height = height 
        self.img_info = img_info
        
        self.path_save_output_xt_to_x0 = path_save_output_xt_to_x0 

        t0 = time.perf_counter() 

        ref_images_raw = self.load_image(ref_images_raw)
        ref_images_raw = ref_images_raw.to(self.device)
        if self.offload:
            self.ae = self.ae.to(self.device)
        ref_images = self.ae.encode(ref_images_raw.to(self.device) * 2 - 1)

        if self.path_save_pt_output is not None:
            torch.save(ref_images.detach().cpu(), os.path.join(self.path_save_pt_output, f"original_x.pt"))

        if self.offload:
            self.ae = self.ae.cpu()
            cudagc()

        seed = int(seed)
        seed = torch.Generator(device="cpu").seed() if seed < 0 else seed

        if init_image is not None:
            init_image = self.load_image(init_image)
            init_image = init_image.to(self.device)
            init_image = torch.nn.functional.interpolate(init_image, (height, width))
            if self.offload:
                self.ae = self.ae.to(self.device)
            init_image = self.ae.encode(init_image.to() * 2 - 1)
            if self.offload:
                self.ae = self.ae.cpu()
                cudagc()
        
        x = torch.randn(
            num_samples,
            16,
            height // 8,
            width // 8,
            device=self.device,
            dtype=torch.bfloat16,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )

        timesteps = sampling.get_schedule(
            num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True
        )

        if init_image is not None:
            t_idx = int((1 - image2image_strength) * num_steps)
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t * x + (1.0 - t) * init_image.to(x.dtype)

        x = torch.cat([x, x], dim=0)


        ref_images = torch.cat([ref_images, ref_images], dim=0)
        ref_images_raw = torch.cat([ref_images_raw, ref_images_raw], dim=0)
        inputs = self.prepare([prompt, negative_prompt], x, ref_image=ref_images, ref_image_raw=ref_images_raw)


        # print("timesteps: ", timesteps) 
        early_timesteps = timesteps[:num_early_steps+1]
        last_timesteps = timesteps[num_early_steps+1:]  
        # print(len(timesteps)) 
        # print(len(early_timesteps))
        # print(len(last_timesteps)) 

        # print("early_timesteps: ", early_timesteps)
        # print("last_timesteps: ", last_timesteps) 

        # print("inputs['img'].shape: ", inputs["img"].shape)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x, pred = self.denoise_original(
                    img=inputs["img"], 
                    img_ids=inputs["img_ids"], 
                    llm_embedding=inputs["llm_embedding"], 
                    txt_ids=inputs["txt_ids"], 
                    timesteps=early_timesteps, 
                    cfg_guidance=cfg_guidance,
                    mask=inputs["mask"], 
                    show_progress=show_progress,
                    timesteps_truncate=1.0,
            )

            img_x0 = x + (0 - early_timesteps[-1]) * pred  

            # ----- Save early-preview image ----- #
            self.get_x_to_vae(img_x0[:, :x.shape[1] // 2], title=f"{seed}-x_t-x_0-to_vae-{num_early_steps}") 

        return {
            "img": x, 
            "img_ids": inputs["img_ids"], 
            "llm_embedding": inputs["llm_embedding"], 
            "txt_ids": inputs["txt_ids"], 
            "timesteps": last_timesteps, 
            "mask": inputs["mask"]
        }