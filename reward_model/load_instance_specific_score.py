# ==================== Import Packages ==================== #
import time
import sys
import os 

sys.path.append(".")

import numpy as np 
import json 

import traceback
import re

from PIL import Image

from reward_model.viescore.mllm_tools.openai import GPT4o
from reward_model.viescore.mllm_tools.qwen25vl_api import QwenVL

# ==================== Constant Parameters ==================== #
prompt_instance_specific_answer_5 = '''System Prompt
You are an “Image-Edit Compliance Judge.”
For every dialogue turn you will receive:
• EDIT_INSTRUCTION – A text description of the desired change.
• QUESTION_LIST – exactly five yes/no questions, each asking whether a certain visual condition is true after the edit.
• ORIGINAL_IMAGE – The initial image before editing.
• EDITED_IMAGE - an edited version of the initial image.

Your task:
Imagine the edit is carried out exactly as written and reason about the resulting image.
For each of the five questions decide “yes” (the condition is satisfied) or “no” (the condition is not satisfied or cannot be inferred). When unsure, answer “no.”
Return nothing except a JSON object with five keys: "Q1", "Q2", "Q3", "Q4", "Q5".
– The value of every key must be the lowercase string "yes" or "no".
– Do not output any explanations, comments, or additional keys.

Output Format:
- Return a single JSON object with the following structure:
{
"Q1": "yes|no",
"Q2": "yes|no",
"Q3": "yes|no",
"Q4": "yes|no",
"Q5": "yes|no"
}

User: 
EDIT_INSTRUCTION: <edit_instruction>
QUESTION_LIST: <instance-specific_question>'''


prompt_instance_specific_answer_1 = '''System Prompt:
You are a meticulous visual-edit evaluator.
The user will supply four textual inputs, in this exact order:
- Edit Instruction: the user’s textual instruction that specifies how the image should be edited.
- Question: The specific, factual question you must answer yes/no.
- Original Image: the image to be edited.
- Edited Image: an edited version of the initial image.

Your task:
- Compare the Original Image with the Edited Image and determine the question yes or no.

Output format: 
- Return one and only one JSON object in the following structure:
{
"answer": "yes" | "no",
"reason": "<brief justification (≤30 words)>"
}

Rules:
- The value of "answer" must be lowercase "yes" or "no" only.
- Provide a concise reason highlighting the key evidence.
- Do not output anything outside the JSON object.

User:
EDIT_INSTRUCTION: <edit_instruction>
QUESTION_LIST: <instance-specific_question>'''

# ==================== Functions ==================== #
def process_instance_specific_score_single_item(image_pairs, edit_instruction, question, model, max_retries=5, **kwargs):

    for retry in range(max_retries):
        try:
            output_dict = {}

            # ----- Alignment evaluation ----- #
            prompt_full = prompt_instance_specific_answer_1.replace("<edit_instruction>", edit_instruction).replace("<instance-specific_question>", question)
            prompt = model.prepare_prompt(image_pairs, prompt_full)
            response = model.get_parsed_output(prompt, constant_flag=True)
            match = re.search(r'{[\s\S]*}', response)

            # print(response)
            
            if match:
                response_clean = match.group(0)
                data = json.loads(response_clean)
            else:
                print(response)
                raise ValueError("No JSON found in response")
            
            output_dict["answer"] = data["answer"]
            output_dict["reason"] = data["reason"]

            
            for temp_key in kwargs:
                output_dict[temp_key] = kwargs[temp_key]

            return output_dict
        
        except Exception as e:
            print("Error occurred:")
            traceback.print_exc()

            if retry < max_retries - 1:
                wait_time = (retry + 1) * 2  # exponential-ish backoff: 2s, 4s, 6s...
                print(f"Error processing (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to process after {max_retries} attempts: {e}")
                return

    return None



def process_instance_specific_score_full_item(image_pairs, edit_instruction, questions, model, max_retries=5, **kwargs):

    for retry in range(max_retries):
        try:
            output_dict = {}

            # ----- Alignment evaluation ----- #
            prompt_full = prompt_instance_specific_answer_5.replace("<edit_instruction>", edit_instruction).replace("<instance-specific_question>", str(questions))
            prompt = model.prepare_prompt(image_pairs, prompt_full)
            response = model.get_parsed_output(prompt, constant_flag=True)
            match = re.search(r'{[\s\S]*}', response)

            # print(response)
            
            if match:
                response_clean = match.group(0)
                data = json.loads(response_clean)
            else:
                print(response)
                raise ValueError("No JSON found in response")
            
            output_dict["answer"] = data

            for temp_key in kwargs:
                output_dict[temp_key] = kwargs[temp_key]

            return output_dict
        
        except Exception as e:
            print("Error occurred:")
            traceback.print_exc()

            if retry < max_retries - 1:
                wait_time = (retry + 1) * 2  # exponential-ish backoff: 2s, 4s, 6s...
                print(f"Error processing (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to process after {max_retries} attempts: {e}")
                return

    return None



# ==================== Main ==================== #
if __name__ == '__main__':
    # ----- Start ----- #
    T_Start = time.time()
    print("Program started!\n")
    print("Python executable: ", sys.executable)
    print("")

    # ---------- step1: Smoke test ---------- #
    # Requires OPENAI_API_KEY (or DASHSCOPE_API_KEY) to be set in the environment.
    model = GPT4o(model_name="gpt-4.1")

    original_img = Image.open("path/to/original_image.png").convert('RGB')
    edit_img = Image.open("path/to/edited_image.png").convert('RGB')

    edit_instruction = "Change the background to a forest."
    image_pairs = [original_img, edit_img]

    questions = [
        "Has the original background behind the subjects been completely replaced with a forest scene?",
        "Does the new forest background extend across the entire area where the previous background was visible?",
        "Are all elements not part of the background (e.g., the man, the dog, table, and food) unchanged in appearance and position?",
        "Does the transition between the subjects (including their outlines) and the new forest background appear natural and seamless, without visible cutout lines or mismatched lighting?",
        "Are there no new visual artifacts, distortions, or color inconsistencies present in the edited image when compared to the original, except for the intended background change?",
    ]

    for temp_question in questions:
        output_dict = process_instance_specific_score_single_item(image_pairs, edit_instruction, temp_question, model)
        print(output_dict)

    # ----- End ----- #
    T_End = time.time()
    T_Sum = T_End - T_Start
    T_Hour = int(T_Sum / 3600)
    T_Minute = int((T_Sum % 3600) / 60)
    T_Second = round((T_Sum % 3600) % 60, 2)
    print("\nExecution Time: {}h {}m {}s".format(T_Hour, T_Minute, T_Second))
    print("Program finished!")
