# ==================== Import Packages ==================== #
import time
import sys
import os 

sys.path.append("edit_model/Step1X_Edit")
sys.path.append("edit_model/FLUX_Kontext")

from pprint import pprint

from PIL import Image  

import numpy as np 
import json 
import random 

import argparse

import logging

from concurrent.futures import ThreadPoolExecutor, as_completed

import torch 
import torch.distributed as dist

from tqdm import tqdm 

# ----- Project imports ----- #
from select_metric.early_stop_util import criterion_early_stop_strategy
from select_metric.select_util import save_json, save_json_cn, setup_logging, set_seed, process_single_item


from utils_ADE_CoT.util_generate_adaptive_nums import get_adaptive_nums 
from utils_ADE_CoT.util_instance_specific_verifier import process_single_item_generate_specific_question 

from reward_model.viescore import VIEScore 
from reward_model.viescore.mllm_tools.openai import GPT4o
from reward_model.viescore.mllm_tools.qwen25vl_api import QwenVL

from edit_model.edit_model_util import get_Step1X_Edit_model, get_FLUX_Kontext_model

from reward_model.load_instance_specific_score import process_instance_specific_score_full_item

# ----- Step1X-Edit imports ----- #
from modules.multigpu import parallel_transformer, teacache_transformer, parallel_teacache_transformer
from inference_Step1X_Edit_util import ImageGenerator, teacache_init, cfg_usp_level_setting

# ----- FLUX Kontext imports ----- #
from pipeline_flux_kontext_modified import FluxKontextPipeline

import warnings
warnings.filterwarnings("ignore")

# ==================== Constant Parameters ==================== #
generate_way = "Baseline_seed{}-1024" 


# ==================== Functions ==================== #



# ==================== Main ==================== #
if __name__ == '__main__':
    # ----- Start ----- #
    T_Start = time.time()
    print("Program started!\n")
    print("Python executable: ", sys.executable)
    print("")

    # ---------- step0: CLI parsing ---------- # 
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_json_dir', type=str, required=True, help='Path to the intput') 
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output image directory')

    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation')

    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--model_name', type=str, default='step1x_edit', choices=['step1x_edit', 'flux_kontext'])
    
    parser.add_argument('--num_samples', default=32, type=int)

    parser.add_argument('--try_times', default=3, type=int)
     
    parser.add_argument('--logging_str', type=str, default=None)
    
    parser.add_argument('--num_steps', type=int, default=28, help='Number of diffusion steps')

    parser.add_argument("--mllm_backbone", type=str, default="qwen-vl-max", choices=["gpt4o", "gpt4.1", "qwen25vl", "qwen-vl-max", "qwen3-vl-plus", "qwen2.5-vl-7b-instruct"])

    parser.add_argument('--max_workers', default=16, type=int) 

    # ----- Final-score aggregation ----- # 
    parser.add_argument("--final_score_aggregate_way", type=str, default="vie-specific", choices=["vie", "vie-specific", "vie-hq", "vie-specific-hq"]) 


    # ============================================================ # 
    #  Model configuration
    # ============================================================ #   
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint') 
    parser.add_argument('--cfg_guidance', type=float, default=6.0, help='CFG guidance strength')
    parser.add_argument('--size_level', default=512, type=int)

    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--enable_cudagc', action='store_true', help='enable cudagc()')

    # ----- Step1X-Edit options ----- #
    parser.add_argument('--offload', action='store_true', help='Use offload for large models')
    parser.add_argument('--quantized', action='store_true', help='Use fp8 model weights')
    parser.add_argument('--lora', type=str, default=None)
    parser.add_argument('--ring_degree', type=int, default=1)
    parser.add_argument('--ulysses_degree', type=int, default=1)
    parser.add_argument('--cfg_degree', type=int, default=1)
    parser.add_argument('--teacache', action='store_true')
    parser.add_argument('--teacache_threshold', type=float, default=0.2, help='Used to control the acceleration ratio of teacache')
    parser.add_argument('--prefix_way', type=str, default="default") 

    # ============================================================ # 
    #  Init: enabled strategies
    # ============================================================ #  
    parser.add_argument("--early_stop_strategy", type=str, default=None,) # adaptive_TTS_nums-early_prune_rank-adaptive_stop
    
    # ----- Early-strategy options ----- #
    parser.add_argument("--early_sample_generate_way", type=str, default="xt_to_x0", choices=["xt_to_x0"]) # "small_steps", 
    parser.add_argument('--small_steps_name', type=str, default="_xt_to_x0")

    # ----- Centroid selection ----- #
    parser.add_argument('--centroid_select_way', type=str, default="clip", choices=["clip", "dino"])

    # ============================================================ # 
    #  Strategy 1: Adaptive Sampling
    # ============================================================ #  
    parser.add_argument('--Adaptive_TTS_nums_flag', action='store_true') 

    # ============================================================ # 
    #  Strategy 2: Early Pruning and Ranking
    # ============================================================ # 
    # ----- Preview mechanism ----- #
    parser.add_argument('--num_early_steps', type=int, default=4, help='Number of diffusion steps') 
    parser.add_argument('--xt_to_x0_early_key_name', type=str, default="x_t-x_0-to_vae-4")  # auto-set

    # ----- Pruning ----- #
    parser.add_argument('--early_prune_flag', action='store_true')
    parser.add_argument('--prune_score_way', type=str, default="vie-caption-region") 
    parser.add_argument('--reject_score', default=0, type=float)   
    parser.add_argument('--mllm_delete_retain_num', default=4, type=int) 

    # ----- Similarity filter ----- #
    parser.add_argument('--sim_remove_flag', action='store_true')
    parser.add_argument('--feat_crop_thred', type=float, default=0.96)
    parser.add_argument('--feat_diff_crop_thred', type=float, default=0.9)
    parser.add_argument('--mean_crop_thred', type=float, default=0.95)

    # ----- Ranking ----- #
    parser.add_argument('--descend_rank_falg', action='store_true') 

    # ============================================================ # 
    #  Strategy 3: Adaptive Stopping
    # ============================================================ # 

    # ----- Preview mechanism ----- #
    parser.add_argument('--num_late_steps', type=int, default=20, help='Number of diffusion steps')   
    parser.add_argument('--xt_to_x0_late_key_name', type=str, default="x_t-x_0-to_vae-20") # auto-set

    # ----- Late retain ----- #
    parser.add_argument('--late_retain_flag', action='store_true')
    parser.add_argument('--retain_score_way', type=str, default="vie-caption-region")  
    parser.add_argument('--retain_score_adaptive_thred', type=float, default=1)
    parser.add_argument('--mllm_late_retain_num', default=0, type=int) 

    
    # ----- High-confidence stop ----- #
    parser.add_argument('--high_confidence_stop_flag', action='store_true')
    parser.add_argument("--high_confidence_score_way", type=str, default="semantic_overall_specific", choices=["semantic_overall", "semantic_overall_specific"]) # high_confidence_score_way
    parser.add_argument('--confi_VIEscore_thred', type=float, default=7.99)
    parser.add_argument('--confi_Semantic_thred', type=float, default=7.99)
    parser.add_argument('--confi_HQ_thred', type=float, default=10)
    parser.add_argument('--confi_instance_specific_thred', type=float, default=4)
    parser.add_argument('--high_confi_num', default=1, type=int)

    # ----- Instance-Specific verifier ----- #
    parser.add_argument('--instance_specific_key', type=str, default="gpt4_1_w_example", choices=["gpt4_1", "gpt4_1_w_example"])
    parser.add_argument("--instance_specific_backbone", type=str, default="qwen-vl-max", choices=["gpt4o", "gpt4.1", "qwen25vl", "qwen-vl-max", "qwen2.5-vl-7b-instruct", "qwen3-vl-plus", "qwen2.5-vl-72b-instruct"])
    parser.add_argument('--instance_specific_exp_name', type=str, default="") 
    parser.add_argument('--lambda_instance_specific', type=float, default=0.1)

    # ----- Global-score backbone ----- #
    parser.add_argument("--global_score_backbone", type=str, default="qwen-vl-max", choices=["gpt4o", "gpt4.1", "qwen25vl", "qwen-vl-max", "qwen2.5-vl-7b-instruct", "qwen3-vl-plus", "qwen2.5-vl-72b-instruct"])
    
    # ============================================================ # 
    #  Verifier configuration
    # ============================================================ # 

    # ----- MLLM scoring ----- #
    parser.add_argument("--mllm_delete_score_way", type=str, default="semantic_overall", choices=["semantic", "overall", "semantic_overall"])
    
    # ----- Edited-region scoring ----- #
    parser.add_argument("--edited_region_backbone", type=str, default="qwen-vl-max", choices=["gpt4o", "gpt4.1", "qwen25vl", "qwen-vl-max", "qwen2.5-vl-7b-instruct"])  
    parser.add_argument('--lambda_region', type=float, default=1)

    parser.add_argument('--remove_sim_threthd', type=float, default=1)
    
    # ----- Caption scoring ----- #
    parser.add_argument('--caption_min_clip_sim', type=float, default=0.27) 
    parser.add_argument("--caption_backbone", type=str, default="gpt4o", choices=["gpt4o", "gpt4.1", "qwen25vl", "qwen-vl-max", "qwen2.5-vl-7b-instruct"])
    parser.add_argument('--caption_exp_name', type=str, default="exp1")
    parser.add_argument('--lambda_caption', type=float, default=3)
    
    # ============================================================ # 
    #  Anchor configuration
    # ============================================================ # 
    # # ----- Correct-anchor settings ----- #
    # parser.add_argument('--anchor_VIEscore_thred', type=float, default=8)
    # parser.add_argument('--anchor_Semantic_thred', type=float, default=8)
    # parser.add_argument('--anchor_retain_num', type=int, default=16)
    
    args = parser.parse_args()

    args.num_steps = 28

    if args.model_name != "step1x_edit":
        generate_way = f"{args.model_name}-{generate_way}" 

    # ---------- step0.25: Build editing model ---------- # 
    if args.model_name.lower() == "step1x_edit":
        model_image_edit = get_Step1X_Edit_model(args)
    elif args.model_name.lower() == "flux_kontext":
        model_image_edit = get_FLUX_Kontext_model(args)

    # ----- Multi-GPU init ----- #
    print("\nMulti-GPU init ...")
    rank = dist.get_rank()
    args.local_rank = rank

    world_size = dist.get_world_size()
    args.world_size = world_size

    print("args.local_rank: ", args.local_rank)
    print("args.world_size: ", args.world_size)

    # ---------- step0.5: Build seed list ---------- # 
    seed = args.seed
    set_seed(seed)

    exp_dict = {}
    for idx_exp in range(args.try_times):
        # num_samples unique random seeds for this experiment
        seed_list = random.sample(range(0, 65535), args.num_samples - 1) 
        seed_list = [args.seed] + seed_list

        exp_dict[idx_exp] = [generate_way.format(seed) for seed in seed_list]


    # ---------- step0.75: Print hyperparameter info ---------- # 
    if args.early_stop_strategy is not None:
        early_stop_strategy_list = args.early_stop_strategy.split("-")
        print("\n============================================================")
        print("early_stop_strategy_list: ", early_stop_strategy_list) 
        print("============================================================")

        if "adaptive_TTS_nums" in early_stop_strategy_list:
            print("Enabling Adaptive Sampling strategy ...")
            args.Adaptive_TTS_nums_flag = True 

        if "early_prune_rank" in early_stop_strategy_list: 
            print("Enabling Early Pruning and Ranking strategy ...")

            args.early_prune_flag = True  
            args.sim_remove_flag = True  
            args.descend_rank_falg = True  

        if "adaptive_stop" in early_stop_strategy_list: 
            print("Enabling Adaptive Stopping strategy ...")

            args.late_retain_flag = True  
            args.high_confidence_stop_flag = True   


    print("\n============================================================")
    print("Effective configuration: ")
    print("============================================================")

    if args.Adaptive_TTS_nums_flag: 
        print("------------------------------------------") 
        print("Adaptive_TTS_nums_flag")   

    if args.early_prune_flag: 
        print("------------------------------------------")  
        print("early_prune_flag") 
        print("\tprune_score_way: ", args.prune_score_way) 
        print("\tnum_early_steps: ", args.num_early_steps) 
        print("\treject_score: ", args.reject_score) 
        print("\tmllm_delete_retain_num: ", args.mllm_delete_retain_num)  

        args.xt_to_x0_early_key_name = f"x_t-x_0-to_vae-{args.num_early_steps}"

    if args.sim_remove_flag:  
        print("------------------------------------------")   
        print("sim_remove_flag")  
        print("\tfeat_crop_thred: ", args.feat_crop_thred) 
        print("\tfeat_diff_crop_thred: ", args.feat_diff_crop_thred) 
        print("\tmean_crop_thred: ", args.mean_crop_thred) 

        args.xt_to_x0_early_key_name = f"x_t-x_0-to_vae-{args.num_early_steps}"


    if args.descend_rank_falg: 
        print("------------------------------------------")  
        print("descend_rank_falg")  

    if args.late_retain_flag: 
        print("------------------------------------------")  
        print("late_retain_flag")  

        print("\tretain_score_way: ", args.retain_score_way)  
        print("\tnum_late_steps: ", args.num_late_steps)  
        print("\tretain_score_adaptive_thred: ", args.retain_score_adaptive_thred)

        args.xt_to_x0_late_key_name = f"x_t-x_0-to_vae-{args.num_late_steps}"  

    if args.high_confidence_stop_flag:  
        print("------------------------------------------")   
        print("high_confidence_stop_flag")   

        print("\thigh_confidence_score_way: ", args.high_confidence_score_way)   
        print("\thigh_confi_num: ", args.high_confi_num) 

    # ---------- step1: Build runtime configuration ---------- # 
    print("\nBuilding runtime configuration ...")

    # ----- Final scorer ----- #
    vie_score_gpt4 = VIEScore(backbone="gpt4.1", task="tie") 

    # ----- Global scorer ----- #
    vie_score_global = VIEScore(backbone=args.global_score_backbone, task="tie") 


    # ----- Build metrics ----- #
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    criterion = criterion_early_stop_strategy(device) 

    # ----- Load run info ----- #
    with open(args.input_json_dir, "r") as f:
        data_input = json.load(f) 
 
    # ---------- step2: Run per-case pipeline ---------- # 
    for path_input_image in data_input:

        # ----- Build output paths ----- #
        path_output = os.path.join(args.output_dir, args.model_name, path_input_image.split("/")[-1]).replace(".png", "").replace(".jpg", "")
        os.makedirs(path_output, exist_ok=True) 

        path_output_image_final = os.path.join(path_output, "final_image")
        os.makedirs(path_output_image_final, exist_ok=True) 
        
        path_output_image_early = os.path.join(path_output, "xt_to_x0")    
        os.makedirs(path_output_image_early, exist_ok=True)  

        path_output_pt_output = os.path.join(path_output, "pt_output") 
        os.makedirs(path_output_pt_output, exist_ok=True)  
    
        log_file = os.path.join(path_output, f"log.txt") 
        if os.path.exists(log_file):
            os.remove(log_file)
        setup_logging(log_file)

        logging.info(f"Processing path_input_image: {path_input_image}")

        # ----- Load case info ----- #
        instruction = data_input[path_input_image]["instruction"] 
        logging.info(f"instruction: {instruction}")  

        input_image = Image.open(path_input_image).convert("RGB")  

        if "caption" in args.prune_score_way:   
            original_caption = data_input[path_input_image]["original_caption"] 
            edited_caption = data_input[path_input_image]["edited_caption"]  

            logging.info(f"original_caption: {original_caption}")  
            logging.info(f"edited_caption: {edited_caption}")  

        if "region" in args.prune_score_way: 
            mask_path = data_input[path_input_image]["mask_path"]  

            logging.info(f"mask_path: {mask_path}")  

        # ----- Init record dict ----- #
        result_dict_all = {}

        # ----- Early stage ----- #
        data_early_VIEscore = None 
        data_early_caption_score = None 
        data_early_region_score = None 
        if args.early_prune_flag or args.sim_remove_flag: 
            data_early_VIEscore = {} 

            if "caption" in args.prune_score_way:  
                data_early_caption_score = {} 

            if "region" in args.prune_score_way: 
                data_early_region_score = {} 

        # ----- Late stage ----- #
        data_late_VIEscore = None  
        data_late_caption_score = None 
        data_late_region_score = None  
        if args.late_retain_flag:  
            data_late_VIEscore = {} 

            if "caption" in args.retain_score_way:  
                data_late_caption_score = {} 

            if "region" in args.retain_score_way: 
                data_late_region_score = {} 

        # ----- Final score setup ----- #
        data_final_VIEscore = {}  
        data_instance_specific_score = None 
        model_instance_specific = None 
        instance_specific_questions = None 
        if "specific" in args.final_score_aggregate_way:
            data_instance_specific_score = {}   

            if args.instance_specific_backbone == "gpt4o":
                model_instance_specific = GPT4o(model_name="gpt-4o-0806") 
            elif args.instance_specific_backbone == "gpt4.1": 
                model_instance_specific = GPT4o(model_name="gpt-41-0414-global") 
            elif args.instance_specific_backbone == "qwen25vl":
                model_instance_specific = QwenVL(model_name="qwen2.5-vl-72b-instruct")
            elif args.instance_specific_backbone in ["qwen2.5-vl-7b-instruct", "qwen-vl-max", "qwen-vl-plus"]: 
                model_instance_specific = QwenVL(model_name=args.instance_specific_backbone)


        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ # 
        #      Gather ADE-CoT info      # 
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ # 

        if "specific" in args.final_score_aggregate_way: 
            if "instance_specific_questions" in data_input[path_input_image]: 
                instance_specific_questions = data_input[path_input_image]["instance_specific_questions"] 
            else:
                instance_specific_questions = process_single_item_generate_specific_question([input_image], instruction, model_instance_specific)["questions"]  
                data_input[path_input_image]["instance_specific_questions"] = instance_specific_questions
                save_json(data_input, args.input_json_dir)

            logging.info("------------------------------------------")  
            logging.info("Running Instance-Specific Verifier strategy ...")
            logging.info(f"Generated questions: {instance_specific_questions}")

        if args.Adaptive_TTS_nums_flag:  

            logging.info("------------------------------------------") 
            logging.info("Running Adaptive Sampling strategy ...")

            TTS_nums, data_final_VIEscore, data_instance_specific_score = get_adaptive_nums(args, model_image_edit, path_output_image_final, path_input_image, instruction, 
                                                                                            vie_score_global, data_final_VIEscore, generate_way, data_instance_specific_score, instance_specific_questions, model_instance_specific)

            logging.info(f"TTS_nums: {TTS_nums}")  

        else:
            TTS_nums = args.num_samples
        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ # 
        #      Run inference      # 
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ # 
        
        for idx_exp in exp_dict:
            
            logging.info("========================================================================") 
            logging.info(f"\tRunning experiment {idx_exp} ...")
            logging.info("========================================================================") 

            task_key_list = exp_dict[idx_exp] 
            task_key_list_copy = task_key_list.copy() 

            logging.info(f"{task_key_list}")  

            result_dict = {}  

            num_task_key = len(task_key_list)  

            NFE_sample = 0  

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #      Adaptive Sampling
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if args.Adaptive_TTS_nums_flag:
                task_key_list = task_key_list[:TTS_nums]  

                if num_task_key > len(task_key_list): 
                    logging.info(f"Adaptive Sampling saved: {num_task_key - len(task_key_list)}")
            
            num_task_key_stage1 = len(task_key_list)  

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #      Early Pruning and Ranking  
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  

            # ------------------------------ # 
            #   Init
            # ------------------------------ # 
            NFE_sample += len(task_key_list) * (args.num_early_steps)  

            early_stage_outputs = {}

            early_stage_outputs["task_key_to_x_t"] = {}


            # ------------------------------ # 
            #   Denoise, generate Early Preview
            # ------------------------------ # 
            for task_key in task_key_list: 

                task_seed = int(task_key.split("_seed")[-1].split("-")[0]) 

                save_output_image_path_early = os.path.join(path_output_image_early, f"{task_seed}-x_t-x_0-to_vae-{args.num_early_steps}.png")   

                if args.model_name.lower() == "step1x_edit": 
                    output_early_stages = model_image_edit.generate_early_stage_image(
                        instruction,
                        negative_prompt="",
                        ref_images=input_image,
                        num_samples=1,
                        num_steps=args.num_steps,
                        cfg_guidance=args.cfg_guidance,
                        seed=task_seed,
                        show_progress=True,
                        size_level=args.size_level,  
                        path_save_output_xt_to_x0=path_output_image_early,
                        num_early_steps = args.num_early_steps 
                    )

                    early_stage_outputs["task_key_to_x_t"][task_key] = output_early_stages["img"]  

                    if "img_ids" not in early_stage_outputs: 
                        early_stage_outputs["img_ids"] = output_early_stages["img_ids"] 
                        early_stage_outputs["llm_embedding"] = output_early_stages["llm_embedding"] 
                        early_stage_outputs["txt_ids"] = output_early_stages["txt_ids"]  
                        early_stage_outputs["timesteps"] = output_early_stages["timesteps"]   
                        early_stage_outputs["mask"] = output_early_stages["mask"]   


                elif args.model_name.lower() == "flux_kontext": 

                    img_info = input_image.size  

                    output_early_stages = model_image_edit.generate_early_stage_image(
                        image=input_image, 
                        prompt=instruction, 
                        num_inference_steps=args.num_steps, 
                        generator=torch.Generator().manual_seed(task_seed), 
                        path_save_output_xt_to_x0=path_output_image_early,
                        num_early_steps=args.num_early_steps, 
                        # guidance_scale=args.cfg_guidance, # use default for now
                        seed=task_seed 
                    )

                    early_stage_outputs["task_key_to_x_t"][task_key] = output_early_stages["latents"] 

                    if "image_latents" not in early_stage_outputs: 
                        for key in output_early_stages:  
                            if key != "latents": 
                                early_stage_outputs[key] = output_early_stages[key]

            # ------------------------------ # 
            #   Score Early Preview
            # ------------------------------ # 
            # ----- VIE Score ----- #  
            data_early_VIEscore = {}

            feature_to_key_dict = {}
            output_image_list = []
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                for task_key in task_key_list:

                    task_seed = int(task_key.split("_seed")[-1].split("-")[0]) 

                    save_output_image_path_early = os.path.join(path_output_image_early, f"{task_seed}-x_t-x_0-to_vae-{args.num_early_steps}.png") 
                    output_image_list.append(save_output_image_path_early)

                    feature = executor.submit(process_single_item, input_image, save_output_image_path_early, instruction, vie_score_global) 
                    feature_to_key_dict[feature] = task_key 

            if len(feature_to_key_dict) == 0:
                continue

            for feature, task_key in feature_to_key_dict.items():
                score_dict = feature.result()
                data_early_VIEscore[task_key] = {}
                data_early_VIEscore[task_key]["sementics_score"] = score_dict["sementics_score"]
                data_early_VIEscore[task_key]["quality_score"] = score_dict["quality_score"]
                data_early_VIEscore[task_key]["overall_score"] = score_dict["overall_score"]

            print(data_early_VIEscore) 

            # ----- Caption Score ----- #
            data_early_caption_score = None 
            if "caption" in args.prune_score_way and original_caption is not None and edited_caption is not None: 
                data_early_caption_score = {}

                sim_ti_output, cos_direction_clip = criterion.get_score_by_input_and_output_caption(input_image, output_image_list, original_caption, edited_caption, num_task_key=len(output_image_list), return_score_flag=True) 

                for idx_task in range(len(output_image_list)):
                    task_key = task_key_list[idx_task] 
                    data_early_caption_score[task_key] = {}
                    data_early_caption_score[task_key]["sim_ti"] = sim_ti_output[idx_task].item()
                    data_early_caption_score[task_key]["cos_direction"] = cos_direction_clip[idx_task].item()

            print(data_early_caption_score)

            # ----- Region Score ----- #
            data_early_region_score = None  
            if "region" in args.prune_score_way and mask_path is not None: 
                data_early_region_score = {}

                score_list = criterion.get_score_by_edited_region(input_image, output_image_list, mask_path) 

                if len(set(score_list)) == 1: 
                    data_early_region_score["same_flag"] = True
                else:
                    data_early_region_score["same_flag"] = False

                for idx_task in range(len(output_image_list)):
                    task_key = task_key_list[idx_task] 
                    data_early_region_score[task_key] = {}
                    data_early_region_score[task_key]["score"] = score_list[idx_task]

            print(data_early_region_score)

            # ----- Record per-task score ----- #
            score_early_list = [] 

            for task_key in task_key_list: 

                vie_info = data_early_VIEscore[task_key]  

                score = 0 

                semantic = np.mean(vie_info['sementics_score']) 
                overall_v = np.mean(vie_info['overall_score']) 

                if args.mllm_delete_score_way == "semantic_overall": 
                    score += (semantic + overall_v) / 2 
                elif args.mllm_delete_score_way == "overall":
                    score += overall_v
                elif args.mllm_delete_score_way == "semantic":    
                    score += semantic

                if data_early_caption_score is not None: 
                    score += data_early_caption_score[task_key]["sim_ti"] * args.lambda_caption 

                if data_early_region_score is not None: 
                    score += data_early_region_score[task_key]["score"] * args.lambda_region 

                score_early_list.append(score)
                    
            scores_early = np.array(score_early_list) 

            # ------------------------------ # 
            #   Early Pruning
            # ------------------------------ # 
            if args.early_prune_flag:
                # ----- Delete ----- #
                keep_mask = scores_early > args.reject_score  
                # Pad up if not enough kept
                if keep_mask.sum() < args.mllm_delete_retain_num: 
                    temp_pad_score = 0
                    while keep_mask.sum() < args.mllm_delete_retain_num:
                        temp_pad_score += 1  

                        if temp_pad_score > 10:
                            break 

                        keep_mask = scores_early > args.reject_score - temp_pad_score    

                task_key_new_list   = list(np.asarray(task_key_list)[keep_mask])
                task_key_list = [str(task_key) for task_key in task_key_new_list]  
                scores_early = scores_early[keep_mask] 
                
                if num_task_key_stage1 > len(task_key_list):
                    print("Early Pruning saved: ", num_task_key_stage1 - len(task_key_list))

            num_task_key_stage2_1 = len(task_key_list)  

            print("early_prune: ", task_key_list)

            # ------------------------------ # 
            #   Filter visually similar candidates
            # ------------------------------ # 
            if args.sim_remove_flag:  

                # ----- Group similar images ----- #
                group_dict, idx_to_group_dict = criterion.judge_similar_image_group(input_image, output_image_list, threshold=args.feat_crop_thred, threshold_diff=args.feat_diff_crop_thred, mean_crop_thred=args.mean_crop_thred, path_mask_image=mask_path) 

                # ----- Remove duplicates ----- #
                if len(group_dict) != len(task_key_list): 

                    task_key_new_list = []
                    scores_early_new_list = []
                    
                    for group_idx in group_dict:
                        temp_idx_task_list = group_dict[group_idx]  

                        temp_task_list = []
                        temp_score_list = []
                        for temp_idx_task in temp_idx_task_list:
                            temp_task_list.append(task_key_list[temp_idx_task])
                            temp_score_list.append(scores_early[temp_idx_task]) 

                        if len(temp_idx_task_list) > 1: 

                            # ----- Pick cluster centroid ----- #
                            max_temp_score = np.max(temp_score_list) 
                            temp_temp_task_list = [] 
                            for temp_idx_score, temp_score in enumerate(temp_score_list):
                                if temp_score >= max_temp_score - args.remove_sim_threthd:
                                    temp_temp_task_list.append(temp_task_list[temp_idx_score])  

                            if len(temp_temp_task_list) == 1:
                                task_key_new_list.append(temp_temp_task_list[0]) 
                                scores_early_new_list.append(temp_score_list[0]) 
                            else:
                                temp_temp_output_image_list = []
                                for task_key in temp_temp_task_list:

                                    task_seed = int(task_key.split("_seed")[-1].split("-")[0]) 

                                    save_output_image_path_early = os.path.join(path_output_image_early, f"{task_seed}-x_t-x_0-to_vae-{args.num_early_steps}.png") 
                                    temp_temp_output_image_list.append(save_output_image_path_early) 
                                
                                if args.centroid_select_way == "clip":
                                    img_features = criterion.encode_batch(temp_temp_output_image_list, criterion.model_clip, criterion.transform_clip, metric="clip_i")
                                elif args.centroid_select_way == "dino":
                                    img_features = criterion.encode_batch(temp_temp_output_image_list, criterion.model_dino, criterion.transform_dino, metric="dino")

                                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                                centroid = img_features.mean(axis=0, keepdims=True)
                                cos_sim = img_features @ centroid.t()  # (N,)
                                center_idx = int(torch.argmax(cos_sim))

                                task_key_new_list.append(temp_temp_task_list[center_idx]) 
                                scores_early_new_list.append(temp_score_list[center_idx])  

                        else:
                            task_key_new_list.append(temp_task_list[0]) 
                            scores_early_new_list.append(temp_score_list[0])  

                    task_key_list = task_key_new_list
                    scores_early = np.array(scores_early_new_list)
    
                if num_task_key_stage2_1 > len(task_key_list):
                    print("Filter visually similar saved: ", num_task_key_stage2_1 - len(task_key_list))

            num_task_key_stage2_2 = len(task_key_list)  

            # ------------------------------ # 
            #   Sorting
            # ------------------------------ # 
            if args.descend_rank_falg: 

                print(task_key_list) 

                order = (-scores_early).argsort(kind="stable")[:]                      
                
                task_key_new_list = []
                for temp_order_i in order: 
                    task_key_new_list.append(task_key_list[temp_order_i])   
                task_key_list = task_key_new_list

                print(task_key_list) 

            
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #      Late Retain - depth-first generation
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  

            # ----- Init bookkeeping ----- #
            best_VIEscore_key_list = []
            best_VIEScore = -float("inf")  
            
            if "specific" in args.final_score_aggregate_way:
                best_instance_specific_list = []
                best_instance_specific_score = -float("inf")  

            if args.high_confidence_stop_flag:
                high_conf_list = []

                for task_key in data_final_VIEscore:
                    
                    overall_VIEscore = np.mean(data_final_VIEscore[task_key]["overall_score"])
                    sementics_score = np.mean(data_final_VIEscore[task_key]["sementics_score"])  

                    if overall_VIEscore > best_VIEScore: 
                        best_VIEScore = overall_VIEscore
                        best_VIEscore_key_list = [task_key]
                    elif overall_VIEscore == best_VIEScore:
                        best_VIEscore_key_list.append(task_key) 

                    if "specific" in args.final_score_aggregate_way:
                        instance_specific_score = data_instance_specific_score[task_key]["score"]  

                        if instance_specific_score > best_instance_specific_score:
                            best_instance_specific_score = instance_specific_score
                            best_instance_specific_list = [task_key]
                        elif instance_specific_score == best_instance_specific_score:
                            best_instance_specific_list.append(task_key)

                    if args.high_confidence_score_way == "semantic_overall":
                        cond = (overall_VIEscore >= args.confi_VIEscore_thred and 
                                # overall_HQscore  >= confi_HQ_thred      and
                                sementics_score  >= args.confi_Semantic_thred)
                    elif args.high_confidence_score_way == "semantic_overall_specific":
                        cond = (overall_VIEscore >= args.confi_VIEscore_thred and 
                                # overall_HQscore  >= confi_HQ_thred      and
                                sementics_score  >= args.confi_Semantic_thred and 
                                instance_specific_score >= args.confi_instance_specific_thred) 
                    else:
                        cond = False 

                    if cond:
                        high_conf_list.append(task_key)  

            else:
                high_conf_list = None     # convenience for downstream checks

            if args.late_retain_flag:
                reject_score_late = 0 

            

            # -------------------------
            # Iterate task keys, depth-first generation
            # -------------------------
            for task_key in task_key_list: 

                task_seed = int(task_key.split("_seed")[-1].split("-")[0]) 

                # -------------------------
                #  Compute late score
                # -------------------------
                if args.late_retain_flag:

                    # -------------------------
                    #  Run generation up to step t_l
                    # -------------------------
                    if args.model_name.lower() == "step1x_edit": 
                        outputs_late_stages = model_image_edit.generate_late_stage_image(
                            img=early_stage_outputs["task_key_to_x_t"][task_key],
                            img_ids=early_stage_outputs["img_ids"],
                            llm_embedding=early_stage_outputs["llm_embedding"],
                            txt_ids=early_stage_outputs["txt_ids"],
                            timesteps=early_stage_outputs["timesteps"],
                            cfg_guidance=args.cfg_guidance,
                            mask=early_stage_outputs["mask"], 
                            num_early_steps=args.num_early_steps,
                            num_late_steps =args.num_late_steps,
                            seed=task_seed,
                        )
                    elif args.model_name.lower() == "flux_kontext":
                        outputs_late_stages = model_image_edit.generate_late_stage_image(
                            # Core state
                            latents=early_stage_outputs["task_key_to_x_t"][task_key],
                            image_latents=early_stage_outputs["image_latents"],
                            latent_ids=early_stage_outputs["latent_ids"],
                            
                            # Text conditioning
                            prompt_embeds=early_stage_outputs["prompt_embeds"],
                            pooled_prompt_embeds=early_stage_outputs["pooled_prompt_embeds"],
                            text_ids=early_stage_outputs["text_ids"],
                            
                            # Negative conditioning
                            negative_prompt_embeds=early_stage_outputs["negative_prompt_embeds"],
                            negative_pooled_prompt_embeds=early_stage_outputs["negative_pooled_prompt_embeds"],
                            negative_text_ids=early_stage_outputs["negative_text_ids"],
                            
                            # CFG & guidance
                            do_true_cfg=early_stage_outputs["do_true_cfg"],
                            true_cfg_scale=early_stage_outputs["true_cfg_scale"],
                            guidance=early_stage_outputs["guidance"],
                            
                            # IP-adapter
                            image_embeds=early_stage_outputs["image_embeds"],
                            negative_image_embeds=early_stage_outputs["negative_image_embeds"],
                            
                            # Timesteps
                            timesteps=early_stage_outputs["timesteps"],
                            begin_index_offset=early_stage_outputs["begin_index_offset"],
                            
                            # Inference params
                            num_inference_steps=args.num_steps,
                            num_late_steps=args.num_late_steps,
                            
                            # Image info
                            height=early_stage_outputs["height"],
                            width=early_stage_outputs["width"],
                            output_type=early_stage_outputs["output_type"],
                            
                            # Preview save
                            path_save_output_xt_to_x0=early_stage_outputs["path_save_output_xt_to_x0"],
                            seed=task_seed,
                            
                            # Optional
                            show_progress=True,
                            image_init_info=early_stage_outputs["image_init_info"],
                        )

                    score = 0

                    save_output_image_path_late = os.path.join(path_output_image_early, f"{task_seed}-x_t-x_0-to_vae-{args.num_late_steps}.png") 

                    vie_score_late = process_single_item(input_image, save_output_image_path_late, instruction, vie_score_global)  

                    score += (vie_score_late["overall_score"] + vie_score_late["sementics_score"]) / 2

                    print(vie_score_late)

                    if "caption" in args.prune_score_way and original_caption is not None and edited_caption is not None: 

                        sim_ti_output, cos_direction_clip = criterion.get_score_by_input_and_output_caption(input_image, [save_output_image_path_late], original_caption, edited_caption, num_task_key=1, return_score_flag=True)

                        # print(sim_ti_output)

                        score += sim_ti_output[0].item() * args.lambda_caption

                    if "region" in args.prune_score_way and mask_path is not None: 
                        score_list = criterion.get_score_by_edited_region(input_image, [save_output_image_path_late], mask_path) 

                        score += score_list[0] * args.lambda_region 

                        # print(score_list)

                    # ----- Adaptive threshold update ----- #
                    if score > reject_score_late: 
                        reject_score_late = score   

                    # Drop low-score samples
                    elif score < reject_score_late - args.retain_score_adaptive_thred:  
                        NFE_sample += args.num_late_steps - args.num_early_steps
                        print("Score too low, late retain dropping task key: ", task_key)
                        continue

                    # ------------------------------ # 
                    #   Normal generation
                    # ------------------------------ #  
                    if args.model_name.lower() == "step1x_edit": 
                        output_image = model_image_edit.generate_final_stage_image(
                            img=outputs_late_stages["img"], 
                            img_ids=early_stage_outputs["img_ids"],
                            llm_embedding=early_stage_outputs["llm_embedding"],
                            txt_ids=early_stage_outputs["txt_ids"],
                            timesteps=outputs_late_stages["timesteps"],
                            cfg_guidance=args.cfg_guidance,
                            mask=early_stage_outputs["mask"], 
                        )
                    elif args.model_name.lower() == "flux_kontext":
                        output_image_dict = model_image_edit.generate_final_stage_image(
                            # Core state
                            latents=outputs_late_stages["latents"],
                            image_latents=early_stage_outputs["image_latents"],
                            latent_ids=early_stage_outputs["latent_ids"],
                            
                            # Text conditioning
                            prompt_embeds=early_stage_outputs["prompt_embeds"],
                            pooled_prompt_embeds=early_stage_outputs["pooled_prompt_embeds"],
                            text_ids=early_stage_outputs["text_ids"],
                            
                            # Negative conditioning
                            negative_prompt_embeds=early_stage_outputs["negative_prompt_embeds"],
                            negative_pooled_prompt_embeds=early_stage_outputs["negative_pooled_prompt_embeds"],
                            negative_text_ids=early_stage_outputs["negative_text_ids"],
                            
                            # CFG & guidance
                            do_true_cfg=early_stage_outputs["do_true_cfg"],
                            true_cfg_scale=early_stage_outputs["true_cfg_scale"],
                            guidance=early_stage_outputs["guidance"],
                            
                            # IP-adapter
                            image_embeds=early_stage_outputs["image_embeds"],
                            negative_image_embeds=early_stage_outputs["negative_image_embeds"],
                            
                            # Timesteps
                            timesteps=outputs_late_stages["timesteps"],
                            begin_index_offset=outputs_late_stages["begin_index_offset"],
                            
                            # Image info
                            height=early_stage_outputs["height"],
                            width=early_stage_outputs["width"],
                            image_init_info=early_stage_outputs["image_init_info"],
                            
                            # Output params
                            return_dict=True,
                            show_progress=True,
                        )
                        
                        output_image = output_image_dict["images"]

                    name_save = generate_way.format(task_seed) 
                    output_image.save(os.path.join(path_output_image_final, f"{name_save}.png"), lossless=True)  

                else:

                    if args.model_name.lower() == "step1x_edit":
                        output_image = model_image_edit.generate_final_stage_image(
                            img=early_stage_outputs["task_key_to_x_t"][task_key], 
                            img_ids=early_stage_outputs["img_ids"],
                            llm_embedding=early_stage_outputs["llm_embedding"],
                            txt_ids=early_stage_outputs["txt_ids"],
                            timesteps=early_stage_outputs["timesteps"],
                            cfg_guidance=args.cfg_guidance,
                            mask=early_stage_outputs["mask"], 
                        )
                    elif args.model_name.lower() == "flux_kontext":
                        output_image_dict = model_image_edit.generate_final_stage_image(
                            # Core state
                            latents=early_stage_outputs["task_key_to_x_t"][task_key],
                            image_latents=early_stage_outputs["image_latents"],
                            latent_ids=early_stage_outputs["latent_ids"],
                            
                            # Text conditioning
                            prompt_embeds=early_stage_outputs["prompt_embeds"],
                            pooled_prompt_embeds=early_stage_outputs["pooled_prompt_embeds"],
                            text_ids=early_stage_outputs["text_ids"],
                            
                            # Negative conditioning
                            negative_prompt_embeds=early_stage_outputs["negative_prompt_embeds"],
                            negative_pooled_prompt_embeds=early_stage_outputs["negative_pooled_prompt_embeds"],
                            negative_text_ids=early_stage_outputs["negative_text_ids"],
                            
                            # CFG & guidance
                            do_true_cfg=early_stage_outputs["do_true_cfg"],
                            true_cfg_scale=early_stage_outputs["true_cfg_scale"],
                            guidance=early_stage_outputs["guidance"],
                            
                            # IP-adapter
                            image_embeds=early_stage_outputs["image_embeds"],
                            negative_image_embeds=early_stage_outputs["negative_image_embeds"],
                            
                            # Timesteps
                            timesteps=early_stage_outputs["timesteps"],
                            begin_index_offset=early_stage_outputs["begin_index_offset"],
                            
                            # Image info
                            height=early_stage_outputs["height"],
                            width=early_stage_outputs["width"],
                            image_init_info=early_stage_outputs["image_init_info"],
                            
                            # Output params
                            return_dict=True,
                            show_progress=True, 
                        )
                        
                        output_image = output_image_dict["images"]

                    name_save = generate_way.format(task_seed) 
                    output_image.save(os.path.join(path_output_image_final, f"{name_save}.png"), lossless=True)  

                # -------------------------
                #  Compute final score
                # -------------------------

                # ----- Compute score ----- #
                save_output_image_path = os.path.join(path_output_image_final, f"{name_save}.png")
                vie_score_final = process_single_item(input_image, save_output_image_path, instruction, vie_score_global)  

                sementics_score = vie_score_final["sementics_score"]
                overall_VIEscore = vie_score_final["overall_score"]

                if overall_VIEscore > best_VIEScore:
                    best_VIEScore = overall_VIEscore 
                    best_VIEscore_key_list = [task_key]
                elif overall_VIEscore == best_VIEScore:
                    best_VIEscore_key_list.append(task_key) 

                # ----- Compute Instance-Specific score ----- #
                instance_specific_score_dict = process_instance_specific_score_full_item([input_image, output_image], instruction, instance_specific_questions, model_instance_specific) 

                instance_specific_score = 0 
                for temp_question in instance_specific_score_dict["answer"]:
                    if instance_specific_score_dict["answer"][temp_question].lower() == "yes":
                        instance_specific_score += 1 

                if "specific" in args.final_score_aggregate_way:

                    if instance_specific_score > best_instance_specific_score:
                        best_instance_specific_score = instance_specific_score
                        best_instance_specific_list = [task_key]
                    elif instance_specific_score == best_instance_specific_score:
                        best_instance_specific_list.append(task_key)

                # ----- Update step count ----- #
                NFE_sample += args.num_steps - args.num_early_steps  

                # ------------------------------ # 
                #   High-confidence stop
                # ------------------------------ # 
                if args.high_confidence_stop_flag:
                    if len(high_conf_list) < args.high_confi_num and task_key not in high_conf_list: 

                        if args.high_confidence_score_way == "semantic_overall":
                            cond = (overall_VIEscore >= args.confi_VIEscore_thred and 
                                    # overall_HQscore  >= confi_HQ_thred      and
                                    sementics_score  >= args.confi_Semantic_thred)
                        elif args.high_confidence_score_way == "semantic_overall_specific":
                            cond = (overall_VIEscore >= args.confi_VIEscore_thred and 
                                    # overall_HQscore  >= confi_HQ_thred      and
                                    sementics_score  >= args.confi_Semantic_thred and 
                                    instance_specific_score >= args.confi_instance_specific_thred) 
                        else:
                            cond = False 

                        if cond: 
                            high_conf_list.append(task_key)  

                    if len(high_conf_list) >= args.high_confi_num:
                        break  

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #      Final image selection
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  

            if args.final_score_aggregate_way == "vie":
                # ----- VIEscore-only best ----- #
                common_list = best_VIEscore_key_list
            elif args.final_score_aggregate_way == "vie-specific":
                # ----- Best of instance_specific_list ----- #
                common_list = list(set(best_instance_specific_list).intersection(best_VIEscore_key_list))
            
            # ----- Fallback ----- #
            if len(common_list) == 0:
                common_list = best_VIEscore_key_list

            # ----- Pick centroid ----- #
            if len(common_list) >= 2:

                temp_img_list = []
                for temp_task_key in common_list:
                    temp_img_list.append(os.path.join(path_output_image_final, f"{temp_task_key}.png")) 
                
                if args.centroid_select_way == "clip":
                    img_features = criterion.encode_batch(temp_img_list, criterion.model_clip, criterion.transform_clip, metric="clip_i")
                elif args.centroid_select_way == "dino":
                    img_features = criterion.encode_batch(temp_img_list, criterion.model_dino, criterion.transform_dino, metric="dino")

                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                centroid = img_features.mean(axis=0, keepdims=True)
                cos_sim = img_features @ centroid.t()  # (N,)
                center_idx = int(torch.argmax(cos_sim))

                select_task_key = common_list[center_idx] 

            else:
                select_task_key = common_list[0]  

            logging.info(f"select_task_key: {select_task_key}")


            # ------------------------------ # 
            #   Compute final score
            # ------------------------------ # 
            final_score_gpt_score = process_single_item(input_image, os.path.join(path_output_image_final, f"{select_task_key}.png"), instruction, vie_score_gpt4)  

            logging.info(f"final_score_gpt_score: {final_score_gpt_score}")

            logging.info("\n\n\n") 
    
    # ----- End ----- #
    T_End = time.time()
    T_Sum = T_End - T_Start
    T_Hour = int(T_Sum / 3600)
    T_Minute = int((T_Sum % 3600) / 60)
    T_Second = round((T_Sum % 3600) % 60, 2)
    print("\nExecution Time: {}h {}m {}s".format(T_Hour, T_Minute, T_Second))
    print("Program finished!")
