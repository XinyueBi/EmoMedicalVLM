import requests
from PIL import Image
import json
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, AutoModelForImageTextToText
import re
import argparse
import os
from tqdm import tqdm
from copy import deepcopy

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_utils import get_dataset
from emotion_prompts import SYSTEM_PROMPT, USER_PROMPTS

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="vindr_test", help="The dataset to run the model on.")
parser.add_argument("--output_file", type=str, default="output/radvlm.jsonl", help="The file to save the model's predictions to.")
parser.add_argument("--emotion", type=str, default="default", help="The emotion category for the user prompt.")
parser.add_argument("--conv_mode", type=str, default="single", choices=["single", "multi"], help="Whether to use single-turn or multi-turn conversation format.")
args = parser.parse_args()

output_dir = Path(args.output_file).parent
os.makedirs(output_dir, exist_ok=True)

if args.emotion == "default" and args.conv_mode == "multi":
    exit(0)
    
import os
import glob
import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, AutoConfig, LlavaOnevisionForConditionalGeneration

def load_radvlm(model_path: str, device: str = None, torch_dtype: str = "auto"):
    """
    Load RadVLM model and processor from a local folder (or Hugging Face repo ID).
    Args:
        model_path: path or repo id with RadVLM weights and configs.
        device: 'cuda', 'cpu', or None to auto-select.
        torch_dtype: one of {'auto', 'float16', 'bfloat16', 'float32'}.
    Returns:
        model, processor, device_str
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, "auto")

    processor = AutoProcessor.from_pretrained(model_path)
    if dtype == "auto":
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, low_cpu_mem_usage=True
        )
    else:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True
        )
    model.to(device)
    model.eval()
    return model, processor, device

model_id = "physionet.org/files/radvlm-model/1.0.0/Model_files"
model, processor, device = load_radvlm(model_id, device="cuda", torch_dtype="float16")

processor.tokenizer.pad_token = processor.tokenizer.eos_token
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

samples = get_dataset(args.dataset, None)

# message    
if args.conv_mode == "single":
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {"type": "text", "text": USER_PROMPTS[args.emotion]
                 + (" Indicate the position of the {abnormality}")},
            ],
        },
    ]
else:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPTS[args.emotion]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": """I'm here to help."""},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {"type": "text", "text": "Indicate the position of the {abnormality}"},
            ],
        },
    ]
    
def extract_all_bboxes(text):
    pattern = r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]"
    matches = re.findall(pattern, text)
    bboxes = [[float(x) for x in m] for m in matches]
    return bboxes if bboxes else None

for sample in tqdm(samples, desc="Processing samples"):
    image = Image.open(sample["image"]).convert("RGB")
    abnormality = sample["class_name"]
    
    sample_messages = deepcopy(messages)
    sample_messages[-1]["content"][0]["image"] = image
    sample_messages[-1]["content"][1]["text"] = sample_messages[-1]["content"][1]["text"].format(abnormality=abnormality)
    
    full_prompt = processor.apply_chat_template(sample_messages, add_generation_prompt=True)

    # Prepare model inputs
    inputs = processor(images=image, text=full_prompt, return_tensors="pt", padding=True).to(
        model.device, torch.float16
    )

    # Generate the response
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=1500, do_sample=False)

    # Decode the output
    full_response = processor.decode(output[0], skip_special_tokens=True)
    response = re.split(r"(user|assistant)", full_response)[-1].strip()
    bboxes = extract_all_bboxes(response)
    
    write_dict = {**sample, "predicted_bboxes": bboxes, "response": response}
    with open(args.output_file, "a") as f:
        f.write(json.dumps(write_dict) + "\n")
