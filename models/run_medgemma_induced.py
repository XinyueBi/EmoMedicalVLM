from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import argparse
import requests
import torch
import json
import os
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_utils import get_dataset
from emotion_prompts import SYSTEM_PROMPT, INDUCED_STORY

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE", help="The dataset to run the model on.")
parser.add_argument("--split", type=str, default="test", help="The split of the dataset to run the model on.")
parser.add_argument("--output_file", type=str, default="output/medgemma.jsonl", help="The file to save the model's predictions to.")
parser.add_argument("--emotion", type=str, default="default", help="The emotion category for the user prompt.")
args = parser.parse_args()

# Create output directory if it doesn't exist
output_dir = Path(args.output_file).parent
os.makedirs(output_dir, exist_ok=True)

# Load model and processor
model_id = "google/medgemma-1.5-4b-it"

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.pad_token = processor.tokenizer.eos_token
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

# Load dataset
samples = get_dataset(args.dataset, args.split)


# Run
for sample in tqdm(samples, desc="Processing samples"):
    image = Image.open(sample["image"]).convert("RGB")
    question = sample["question"]
    
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": INDUCED_STORY[args.emotion]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": """I'm here to help. Please share your question, and I will provide a brief answer."""},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]
    
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
        generation = generation[0][input_len:]
    
    decoded = processor.decode(generation, skip_special_tokens=True).strip()
    
    write_dict = {
        "image": sample["image"],
        "question": question,
        "answer": sample["answer"],
        "model_answer": decoded,
        "emotion": args.emotion,
        "location": sample["location"],
        "modality": sample["modality"],
        "answer_type": sample["answer_type"],
        "content_type": sample["content_type"],
    }
    with open(args.output_file, "a") as f:
        f.write(json.dumps(write_dict) + "\n")
