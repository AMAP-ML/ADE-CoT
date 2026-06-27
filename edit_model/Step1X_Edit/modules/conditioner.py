import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from torchvision.transforms import ToPILImage

import sys 
import os 

import torch.distributed as dist

import re 

import logging

to_pil = ToPILImage()

# ===========================================================================
#  Step1X-Edit prefix prompts
# ===========================================================================

Qwen25VL_7b_PREFIX = '''Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:
- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.
- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\n
Here are examples of how to transform or refine prompts:
- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.
- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\n
Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:
User Prompt:'''


Qwen25VL_7b_PREFIX_NO_Think = '''Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:
- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.
- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\n
Here are examples of how to transform or refine prompts:
- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.
- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\n
Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:
User Prompt: <think>Okay, I have finished thinking.</think> '''


Qwen25VL_7b_PREFIX_Think_Bagel = ''''''


# ===========================================================================
#  Enhanced edit instructions with <think>
# ===========================================================================

Qwen25VL_7b_Think_Bagel = ''''''


Qwen25VL_7b_Think_way1 = '''NOTE: You are an expert image editor. Given a user edit instruction, enhance it:
- If too simple, add key visual details.
- If it requires reasoning, infer the intended edit and describe it concretely.
- If the instruction edits one object but logically affects others, include those additional edits.
- If already clear and detailed, keep it unchanged.\n
You first thinks about the reasoning process in the mind and then provides the user with the answer.
Output your reasoning in <think> </think>, and the final enhanced instruction in <answer> </answer>.
Edit Instruction: '''


Qwen25VL_7b_Think_way2 = '''NOTE: You are an expert image editor. Given a user edit instruction, enhance it:
- If too simple, add key visual details.
- If it requires reasoning, infer the intended edit and describe it concretely.
- If the instruction edits one object but logically affects others, include those additional edits.
- If the image involves multiple subjects but only one requires editing, explicitly clarify which is changed and which remains unchanged.
- If already clear and detailed, keep it unchanged.
- Remain concise, specific, and actionable.

You first think about the reasoning process in the mind and then provide the user with the answer.
Output your reasoning in <think> </think>, and the final enhanced instruction in <answer> </answer>.
Edit Instruction: '''

Qwen25VL_7b_Think_way3 = ''''''


Qwen25VL_7b_Think_way4 = ''''''

QWen25VL_7b_enhance_by_edit_captions = '''You are an expert image editing planner.

Given:
1. An input image,
2. A user-provided edit instruction (which may be vague or incomplete),
3. A caption that describes what the edited image should look like,

Your task is to revise and enhance the edit instruction so that, **when applied to the original image**, it can produce an image that matches the given caption.

Your revised instruction should:
- Be visually grounded (based on the input image),
- Include concrete details,
- Incorporate any additional edits implied by the target caption,
- Remain concise, specific, and actionable.

If the original edit instruction is already sufficient, you may keep it mostly unchanged, but improve clarity and precision.

Output your reasoning in <think> </think>, and the final enhanced instruction in <answer> </answer>.'''


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def split_string(s):
    # Replace fullwidth (CJK) quotes with ASCII ones, then walk the string and
    # surround characters inside quoted spans with fullwidth quotes again.
    s = s.replace("\u201c", '"').replace("\u201d", '"')  # use english quotes
    result = []
    in_quotes = False
    temp = ""

    for idx, char in enumerate(s):
        if char == '"' and idx > 155:
            temp += char
            if not in_quotes:
                result.append(temp)
                temp = ""

            in_quotes = not in_quotes
            continue
        if in_quotes:
            if char.isspace():
                pass  # have space token

            result.append("\u201c" + char + "\u201d")
        else:
            temp += char

    if temp:
        result.append(temp)

    return result


class Qwen25VL_7b_Embedder(torch.nn.Module):
    def __init__(self, model_path, max_length=640, dtype=torch.bfloat16, device="cuda", prefix_way="default"):
        super(Qwen25VL_7b_Embedder, self).__init__()
        self.max_length = max_length

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
        ).to(torch.cuda.current_device())

        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=256 * 28 * 28, max_pixels=324 * 28 * 28
        )

        # ----- Prefix mode ----- #
        self.prefix_way = prefix_way
        self.prefix = Qwen25VL_7b_PREFIX
        self.len_specific = 217

        self.update_prefix_way(prefix_way)  

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def update_prefix_way(self, prefix_way):

        print("\nUpdating prefix way:")

        self.prefix_way = prefix_way

        if "no_think" in prefix_way:
            print("\nAppending <think>Okay, I have finished thinking.</think> to prefix prompt")
            self.prefix = Qwen25VL_7b_PREFIX_NO_Think
        elif "prefix_bagel" in prefix_way:
            self.prefix = Qwen25VL_7b_PREFIX_Think_Bagel
        else:
            self.prefix = Qwen25VL_7b_PREFIX

        if "with_bagel" in prefix_way:
            print("\nUsing two-stage <think> dialogue: Bagel way")
            self.think_prompt = Qwen25VL_7b_Think_Bagel
        elif "with_way1" in prefix_way:
            print("\nUsing two-stage <think> dialogue: way 1")
            self.think_prompt = Qwen25VL_7b_Think_way1
        elif "with_way2" in prefix_way:
            print("\nUsing two-stage <think> dialogue: way 2")
            self.think_prompt = Qwen25VL_7b_Think_way2
        elif "with_way3" in prefix_way:
            print("\nUsing two-stage <think> dialogue: way 3")
            self.think_prompt = Qwen25VL_7b_Think_way3 
        elif "with_way4" in prefix_way:
            print("\nUsing two-stage <think> dialogue: way 4")
            self.think_prompt = Qwen25VL_7b_Think_way4
        else:
            self.think_prompt = None

        if "enhance_by_edit_captions" in prefix_way:
            self.think_prompt = QWen25VL_7b_enhance_by_edit_captions

        if "only_rewrite" in prefix_way:
            print("Using rewrite text only")
            self.new_chat_flag = "only_rewrite" 
        else:
            self.new_chat_flag = "add_rewrite_in_one_txt"


        if "delete_normal" in prefix_way:
            self.delete_normal_flag = True 
        else:
            self.delete_normal_flag = False

        # ----- Compute the length of the prefix prompt ----- #
        messages_specific = [{"role": "user", "content": []}]
        messages_specific[0]["content"].append({"type": "text", "text": f"{self.prefix}"})

        text_specific = self.processor.apply_chat_template(
            messages_specific, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )

        inputs_specific = self.processor(
            text=[text_specific],
            padding=True,
        )

        if self.delete_normal_flag:
            self.len_specific = len(inputs_specific["input_ids"][0])
        else:
            self.len_specific = len(inputs_specific["input_ids"][0]) - 6 # 223 - 217
        print("\nlen_specific: ", self.len_specific)

    def forward(self, caption, ref_images):

        text_list = caption
        embs = torch.zeros(
            len(text_list),
            self.max_length,
            self.model.config.hidden_size,
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
        )
        hidden_states = torch.zeros(
            len(text_list),
            self.max_length,
            self.model.config.hidden_size,
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
        )
        masks = torch.zeros(
            len(text_list),
            self.max_length,
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        input_ids_list = []
        attention_mask_list = []
        emb_list = []

        def split_string(s):
            s = s.replace("\u201c", '"').replace("\u201d", '"').replace("'", '''"''')  # use english quotes
            result = []
            in_quotes = False
            temp = ""

            for idx,char in enumerate(s):
                if char == '"' and idx>155:
                    temp += char
                    if not in_quotes:
                        result.append(temp)
                        temp = ""

                    in_quotes = not in_quotes
                    continue
                if in_quotes:
                    if char.isspace():
                        pass  # have space token

                    result.append("\u201c" + char + "\u201d")
                else:
                    temp += char

            if temp:
                result.append(temp)

            return result

        for idx, (txt, imgs) in enumerate(zip(text_list, ref_images)):

            # ----- Run the <think> stage ----- #
            if self.think_prompt is not None and txt != "":
                
                think_messages = [{"role": "user", "content": []}]
                think_messages[0]["content"].append({"type": "image", "image": to_pil(imgs)})
                think_messages[0]["content"].append({"type": "text", "text": f"{txt}"})
                think_messages[0]["content"].append({"type": "text", "text": f"{self.think_prompt}"})
                
                # Preparation for inference
                think_text = self.processor.apply_chat_template(
                    think_messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
                )

                think_image_inputs, _ = process_vision_info(think_messages)

                think_inputs = self.processor(
                    text=[think_text],
                    images=think_image_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                think_old_inputs_ids = think_inputs.input_ids
                think_text_split_list = split_string(think_text)

                think_token_list = []
                for text_each in think_text_split_list:
                    txt_inputs = self.processor(
                        text=text_each,
                        images=None,
                        videos=None,
                        padding=True,
                        return_tensors="pt",
                    )
                    token_each = txt_inputs.input_ids
                    if token_each[0][0] == 2073 and token_each[0][-1] == 854:
                        token_each = token_each[:, 1:-1]
                        think_token_list.append(token_each)
                    else:
                        think_token_list.append(token_each)

                think_new_txt_ids = torch.cat(think_token_list, dim=1).to("cuda")

                think_new_txt_ids = think_new_txt_ids.to(think_old_inputs_ids.device)

                idx1 = (think_old_inputs_ids == 151653).nonzero(as_tuple=True)[1][0]
                idx2 = (think_new_txt_ids == 151653).nonzero(as_tuple=True)[1][0]
                think_inputs.input_ids = (
                    torch.cat([think_old_inputs_ids[0, :idx1], think_new_txt_ids[0, idx2:]], dim=0)
                    .unsqueeze(0)
                    .to("cuda")
                )
                think_inputs.attention_mask = (think_inputs.input_ids > 0).long().to("cuda")

                think_generated_ids = self.model.generate(
                    input_ids=think_inputs.input_ids,
                    attention_mask=think_inputs.attention_mask,
                    pixel_values=think_inputs.pixel_values.to("cuda"),
                    image_grid_thw=think_inputs.image_grid_thw.to("cuda"),
                    max_new_tokens=1400,
                )

                think_generated_text = self.processor.batch_decode(think_generated_ids, skip_special_tokens=True)[0]
                think_generated_text = think_generated_text.replace("<answer> </answer>", "")

                # ----- Extract the final enhanced prompt ----- #
                think_generated_match_text = re.search(r'<answer>(.*?)</answer>', think_generated_text, re.DOTALL)
                if think_generated_match_text:
                    think_answer = think_generated_match_text.group(1).strip() 
                else:
                    print("\nRegex match failed!")
                    think_answer = think_generated_text.strip()
                think_answer = think_answer.replace("\"", "").replace("\'", "")
                print("----------------------------------------------")
                print("original edit instruction: ", txt)
                print("think_answer: ", think_answer)
                logging.info(f"original edit instruction: {txt}")
                logging.info(f"think_answer: {think_answer}")
                
            messages = [{"role": "user", "content": []}]

            messages[0]["content"].append({"type": "text", "text": f"{self.prefix}"})

            messages[0]["content"].append({"type": "image", "image": to_pil(imgs)})

            if self.think_prompt is not None and txt != "":
                if self.new_chat_flag == "only_rewrite": 
                    messages[0]["content"].append({"type": "text", "text": think_answer})
                elif self.new_chat_flag == "add_rewrite_in_one_txt":
                    txt = txt.rstrip()
                    if not txt.endswith('.'):
                        txt += '.'
                    messages[0]["content"].append({"type": "text", "text": f"{txt} {think_answer}"})
            else:
                messages[0]["content"].append({"type": "text", "text": f"{txt}"})

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )

            old_inputs_ids = inputs.input_ids
            text_split_list = split_string(text)

            token_list = []
            for text_each in text_split_list:
                txt_inputs = self.processor(
                    text=text_each,
                    images=None,
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                )
                token_each = txt_inputs.input_ids
                if token_each[0][0] == 2073 and token_each[0][-1] == 854:
                    token_each = token_each[:, 1:-1]
                    token_list.append(token_each)
                else:
                    token_list.append(token_each)

            new_txt_ids = torch.cat(token_list, dim=1).to("cuda")

            new_txt_ids = new_txt_ids.to(old_inputs_ids.device)

            idx1 = (old_inputs_ids == 151653).nonzero(as_tuple=True)[1][0]
            idx2 = (new_txt_ids == 151653).nonzero(as_tuple=True)[1][0]
            inputs.input_ids = (
                torch.cat([old_inputs_ids[0, :idx1], new_txt_ids[0, idx2:]], dim=0)
                .unsqueeze(0)
                .to("cuda")
            )
            inputs.attention_mask = (inputs.input_ids > 0).long().to("cuda")

            # ----- Generate text ----- #
            if self.prefix_way == "with_self_generate" and txt != "":

                generated_ids = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.pixel_values.to("cuda"),
                    image_grid_thw=inputs.image_grid_thw.to("cuda"),
                    max_new_tokens=1200,
                )
                generate_mask = (generated_ids > 0).long().to("cuda")
                
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=generate_mask,
                    pixel_values=inputs.pixel_values.to("cuda"),
                    image_grid_thw=inputs.image_grid_thw.to("cuda"),
                    output_hidden_states=True,
                )

            else:
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.pixel_values.to("cuda"),
                    image_grid_thw=inputs.image_grid_thw.to("cuda"),
                    output_hidden_states=True,
                )

            emb = outputs["hidden_states"][-1]

            embs[idx, : min(self.max_length, emb.shape[1] - self.len_specific)] = emb[0, self.len_specific:][
                : self.max_length
            ]

            masks[idx, : min(self.max_length, emb.shape[1] - self.len_specific)] = torch.ones(
                (min(self.max_length, emb.shape[1] - self.len_specific)),
                dtype=torch.long,
                device=torch.cuda.current_device(),
            )

        # ----- Return outputs ----- #
        llm_encoder_output_dict = {}
        llm_encoder_output_dict["txt"] = embs
        llm_encoder_output_dict["mask"] = masks

        return llm_encoder_output_dict
