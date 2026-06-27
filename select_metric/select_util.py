# ==================== Import Packages ==================== #
import time
import sys
import os 

import numpy as np 
import json 

from PIL import Image
import megfile

import traceback

import logging

import math 

import random 

import torch 

# ==================== Constant Parameters ==================== #


# ==================== Functions ==================== #
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_json(dict, path_save):
    data_json = json.dumps(dict, indent=4) 
    file = open(path_save, 'w')
    file.write(data_json)
    file.close()

def save_json_cn(dict, path_save):
    data_json = json.dumps(dict, ensure_ascii=False, indent=4) 
    file = open(path_save, 'w')
    file.write(data_json)
    file.close()

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    new_area = width * height
    return int(width), int(height), int(new_area)

def process_single_item(pil_image_raw, edit_img_path, instruction, vie_score, max_retries=10):

    for retry in range(max_retries):
        try:

            pil_image_edited = Image.open(megfile.smart_open(edit_img_path, 'rb')).convert("RGB").resize((pil_image_raw.size[0], pil_image_raw.size[1]))

            text_prompt = instruction
            
            score_list = vie_score.evaluate([pil_image_raw, pil_image_edited], text_prompt)

            sementics_score, quality_score, overall_score, SC_dict, PQ_dict = score_list

            print(f"\nsementics_score: {sementics_score}, quality_score: {quality_score}, overall_score: {overall_score}, instruction: {instruction}")
            
            return {
                "sementics_score": sementics_score,
                "quality_score": quality_score,
                "overall_score": overall_score,
                "SC_dict": SC_dict, 
                "PQ_dict": PQ_dict
            }
        except Exception as e:

            print("Error occurred:")
            traceback.print_exc()

            if retry < max_retries - 1:
                wait_time = (retry + 1) * 2  # exponential-ish backoff: 2s, 4s, 6s...
                print(f"Error processing {edit_img_path} (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to process {edit_img_path} after {max_retries} attempts: {e}")
                return


    return None 



def borda_count(candidates, rank_lists, weights=None):
    """
    Aggregate rankings using the Borda Count method.

    Args:
        candidates (list): All candidates.
        rank_lists (list of lists): Multiple ranked lists.
        weights (list or None): Per-list weight; default 1 for every list.

    Returns:
        list of tuples: (candidate, score) sorted by final ranking.
    """
    num_candidates = len(candidates)
    if weights is None:
        weights = [1] * len(rank_lists)
    
    if len(weights) != len(rank_lists):
        raise ValueError("weights must have the same length as rank_lists.")

    scores = {candidate: 0 for candidate in candidates}

    for i, rank_list in enumerate(rank_lists):
        for rank, candidate in enumerate(rank_list):
            # Rank-1 (rank=0) gets N-1 points, rank-N (rank=N-1) gets 0.
            points = num_candidates - 1 - rank
            scores[candidate] += points * weights[i]

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    return sorted_scores

def borda_count_w_weighted_rank(candidates, rank_lists, weights=None, tie_eps=1e-8):
    """
    Borda Count aggregation with support for ties.

    Args:
        candidates (list): All candidates.
        rank_lists (list): Multiple ranked lists. Each element can be:
            - list[str]: strictly ranked list of candidates (no tie info)
            - list[tuple[str, float]]: (candidate, score) list, sorted by score
              descending, with tied candidates handled jointly.
        weights (list or None): Per-list weight; default 1 for every list.
        tie_eps (float): Tolerance for tie detection.

    Returns:
        list[tuple[str, float]]: (candidate, score) sorted by final ranking.
    """
    num_candidates = len(candidates)
    if weights is None:
        weights = [1] * len(rank_lists)

    if len(weights) != len(rank_lists):
        raise ValueError("weights must have the same length as rank_lists.")

    scores = {candidate: 0 for candidate in candidates}

    for i, rank_list in enumerate(rank_lists):
        if not rank_list:
            continue

        first_item = rank_list[0]
        is_pair_list = isinstance(first_item, (tuple, list)) and len(first_item) == 2

        if is_pair_list:
            # (candidate, score) list with possible ties.
            pair_list = [(cand, float(sc)) for cand, sc in rank_list if cand in scores]
            pair_list.sort(key=lambda x: x[1], reverse=True)

            # Group tied entries.
            groups = []  # list[list[candidate]]
            current_group = [pair_list[0][0]]
            current_score = pair_list[0][1]
            for cand, sc in pair_list[1:]:
                if abs(sc - current_score) <= tie_eps:
                    current_group.append(cand)
                else:
                    groups.append((current_group, current_score))
                    current_group = [cand]
                    current_score = sc
            groups.append((current_group, current_score))

            # Assign averaged Borda points to each tied group.
            next_position = 0
            for group_candidates, _ in groups:
                k = len(group_candidates)
                # Group covers positions next_position .. next_position + k - 1
                # whose Borda points are (N - 1 - pos).
                group_points_sum = 0.0
                for offset in range(k):
                    pos = next_position + offset
                    group_points_sum += max(0, num_candidates - 1 - pos)
                avg_points = group_points_sum / float(k)

                for cand in group_candidates:
                    scores[cand] += avg_points * weights[i]

                next_position += k
        else:
            # Strict ranking without tie info.
            for rank, candidate in enumerate(rank_list):
                if candidate not in scores:
                    continue
                points = max(0, num_candidates - 1 - rank)
                scores[candidate] += points * weights[i]

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    return sorted_scores


