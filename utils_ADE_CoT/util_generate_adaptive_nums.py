# ==================== Import Packages ==================== #
import time
import sys
import os 

import torch  

import numpy as np 
import json 

from PIL import Image 

from tqdm import tqdm 

import logging

from edit_model.FLUX_Kontext.inference_FLUX_Kontext_util import process_single_edit_item_by_FLUX_Kontext

from select_metric.select_util import save_json, process_single_item
from reward_model.load_instance_specific_score import process_instance_specific_score_full_item


# ==================== Constant Parameters ==================== #

# ==================== Functions ==================== #

def get_adaptive_nums(args, model_image_edit, output_dir, path_input_image, instruction, vie_score_global, 
                      data_final_VIEscore, generate_way, data_instance_specific_score=None, instance_specific_questions=None, model_instance_specific=None, repeat_times=1):  
    """
    Compute the adaptive sample budget (Strategy 1).
    """

    # ---------- step1: Generate image ---------- #
    with torch.no_grad():

        input_image = Image.open(path_input_image).convert("RGB")
        
        save_path_fullset_source_image = os.path.join(output_dir, "source.png") 
        if not os.path.exists(save_path_fullset_source_image):
            input_image.save(save_path_fullset_source_image) 

        name_save = generate_way.format(args.seed) 
        save_path_fullset = os.path.join(output_dir, f"{name_save}.png")    

        if not os.path.exists(save_path_fullset): 

            start_time = time.time()

            if args.model_name == "step1x_edit":
                output_image = model_image_edit.generate_image(
                    instruction,
                    negative_prompt="",
                    ref_images=input_image,
                    num_samples=1,
                    num_steps=args.num_steps,
                    cfg_guidance=args.cfg_guidance,
                    seed=args.seed,
                    show_progress=True,
                    size_level=args.size_level,
                )[0]
            elif args.model_name.lower() == "flux_kontext":
                output_image = process_single_edit_item_by_FLUX_Kontext(
                    model_image_edit,
                    instruction=instruction,
                    ref_image=input_image,
                    num_steps=args.num_steps,
                    guidance_scale=args.cfg_guidance,
                    seed=args.seed,
                    size_level=args.size_level,
                )
            
            if args.enable_cudagc:
                cudagc()

            print(f"Time taken: {time.time() - start_time:.2f} seconds") 

            output_image.save(save_path_fullset, lossless=True)  

        else:
            output_image = Image.open(save_path_fullset).convert("RGB")   

    # ---------- step2: Compute score ---------- #
    data_final_VIEscore[name_save] = {}  
    data_final_VIEscore[name_save]["sementics_score"] = []
    data_final_VIEscore[name_save]["quality_score"] = [] 
    data_final_VIEscore[name_save]["overall_score"] = []

    
    for i in range(repeat_times):
        score_dict = process_single_item(input_image, save_path_fullset, instruction, vie_score_global) 

        data_final_VIEscore[name_save]["sementics_score"].append(score_dict["sementics_score"])
        data_final_VIEscore[name_save]["quality_score"].append(score_dict["quality_score"])
        data_final_VIEscore[name_save]["overall_score"].append(score_dict["overall_score"])

    if instance_specific_questions is not None:
        instance_specific_score_dict = process_instance_specific_score_full_item([input_image, output_image], instruction, instance_specific_questions, model_instance_specific) 
        data_instance_specific_score[name_save] = instance_specific_score_dict  

        specific_score = 0 
        for temp_key in data_instance_specific_score[name_save]["answer"]:
            if data_instance_specific_score[name_save]["answer"][temp_key].lower() == "yes":
                specific_score += 1 

        data_instance_specific_score[name_save]["score"] = specific_score  

    # ---------- step3: Compute the adaptive sample count ---------- #
    initial_score = np.mean(data_final_VIEscore[name_save]["overall_score"]) 
    if data_instance_specific_score[name_save]["score"] >= 5:
        initial_score = initial_score + 0.1 * data_instance_specific_score[name_save]["score"]


    TTS_nums = calculate_adaptive_budget(initial_score, args.num_samples)  

    return TTS_nums, data_final_VIEscore, data_instance_specific_score 


def calculate_adaptive_budget(S, N, S_max=10.0, N_min=1, gamma=0.12):
    """
    Compute the adaptive sample budget.

    Args:
        S: initial score (edit difficulty proxy)
        S_max: maximum possible score
        N_min: minimum budget
        N: original budget
        gamma: sensitivity hyperparameter

    Returns:
        N_a: adaptive budget
    """
    # Clamp S to S_max to avoid taking a fractional power of a negative number.
    S = min(S, S_max)

    # Ensure the base is non-negative.
    base = max(0, 1 - S / S_max)

    N_a = N_min + round((N - N_min) * (base ** gamma))

    return N_a


def cudagc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()