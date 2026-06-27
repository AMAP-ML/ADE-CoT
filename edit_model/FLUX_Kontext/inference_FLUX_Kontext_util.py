# ==================== Import Packages ==================== #
import time
import sys
import os 

import numpy as np 
import json 

import torch 

from PIL import Image, ImageOps


# ==================== Constant Parameters ==================== #


# ==================== Functions ==================== #

def process_single_edit_item_by_FLUX_Kontext(model_image_edit, instruction, ref_image, num_steps, guidance_scale, seed, size_level, path_save_pt_output=None, path_save_output_xt_to_x0=None):
    """Generate an edited image with the FLUX Kontext model."""

    # Ensure input image is RGB (compose RGBA over white, convert others directly).
    if isinstance(ref_image, Image.Image):
        if ref_image.mode != 'RGB':
            if ref_image.mode == 'RGBA':
                background = Image.new('RGB', ref_image.size, (255, 255, 255))
                background.paste(ref_image, mask=ref_image.split()[3])  # use alpha as mask
                ref_image = background
            else:
                ref_image = ref_image.convert('RGB')

    img_info = ref_image.size

    output_image = model_image_edit(
        image=ref_image,
        prompt=instruction,
        # guidance_scale=guidance_scale, 
        num_inference_steps=num_steps, 
        generator=torch.Generator().manual_seed(seed),
        path_save_pt_output=path_save_pt_output, 
        path_save_output_xt_to_x0=path_save_output_xt_to_x0
    ).images[0]
    
    output_image = output_image.resize(img_info)

    return output_image

