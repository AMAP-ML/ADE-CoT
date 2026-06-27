# ==================== Import Packages ==================== #
import sys
import os

sys.path.append("edit_model/Step1X_Edit")
sys.path.append("edit_model/FLUX_Kontext")

import torch
import torch.distributed as dist

# ----- Step1X-Edit imports ----- #
from modules.multigpu import parallel_transformer, teacache_transformer, parallel_teacache_transformer
from inference_Step1X_Edit_util import ImageGenerator, teacache_init, cfg_usp_level_setting

# ----- FLUX Kontext imports ----- #
from pipeline_flux_kontext_modified import FluxKontextPipeline


# ==================== Functions ==================== #


def get_Step1X_Edit_model(args):
    """Build the Step1X-Edit model."""

    mode = "flash" if args.ring_degree * args.ulysses_degree * args.cfg_degree == 1 else "xdit"

    model_image_edit = ImageGenerator(
        ae_path=os.path.join(args.model_path, 'vae.safetensors'),
        dit_path=os.path.join(args.model_path, "step1x-edit-i1258.safetensors"),
        qwen2vl_model_path=os.path.join(args.model_path, 'Qwen2.5-VL-7B-Instruct'),
        max_length=640,
        quantized=args.quantized,
        offload=args.offload,
        lora=args.lora,
        mode=mode,
        prefix_way=args.prefix_way,
    )

    if args.teacache:
        teacache_init(model_image_edit, args)
        if args.ring_degree * args.ulysses_degree * args.cfg_degree != 1:
            cfg_usp_level_setting(args.ring_degree, args.ulysses_degree, args.cfg_degree)
            parallel_teacache_transformer(model_image_edit)
        else:
            teacache_transformer(model_image_edit)
    else:
        if args.ring_degree * args.ulysses_degree * args.cfg_degree != 1:
            cfg_usp_level_setting(args.ring_degree, args.ulysses_degree, args.cfg_degree)
            parallel_transformer(model_image_edit)

    return model_image_edit


def get_FLUX_Kontext_model(args):
    """Build the FLUX Kontext model."""

    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    model_flux_kontext = FluxKontextPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model_flux_kontext.to("cuda")

    return model_flux_kontext
