import argparse
import torch
import json
import os
from tqdm import tqdm
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "LLaVA-Med"))

from data_utils import get_dataset
from emotion_prompts import USER_PROMPTS_MAIN

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE", help="The dataset to run the model on.")
parser.add_argument("--split", type=str, default="test", help="The split of the dataset to run the model on.")
parser.add_argument("--output_file", type=str, default="output/llavamed.jsonl", help="The file to save the model's predictions to.")
parser.add_argument("--emotion", type=str, default="default", help="The emotion category for the user prompt.")
parser.add_argument("--model_path", type=str, default="microsoft/llava-med-v1.5-mistral-7b", help="LLaVA-Med model path.")
parser.add_argument("--model_base", type=str, default=None, help="Optional base model path.")
parser.add_argument("--conv_mode", type=str, default="mistral_instruct", help="Conversation template mode.")
parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling.")
parser.add_argument("--num_beams", type=int, default=1, help="Number of beams.")
parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum generated tokens.")
parser.add_argument("--max_samples", type=int, default=None, help="Optional limit on number of samples to process.")
args = parser.parse_args()

# Create output directory if it doesn't exist
output_dir = Path(args.output_file).parent
os.makedirs(output_dir, exist_ok=True)

# Load model
disable_torch_init()
model_name = Path(args.model_path).name
tokenizer, model, image_processor, context_len = load_pretrained_model(
    args.model_path,
    args.model_base,
    model_name,
)

device = next(model.parameters()).device

# Load dataset
samples = get_dataset(args.dataset, args.split)
if args.max_samples is not None:
    samples = samples[:args.max_samples]

# Run
for sample in tqdm(samples, desc="Processing samples"):
    image = Image.open(sample["image"]).convert("RGB")
    question = sample["question"]

    user_prompt = USER_PROMPTS_MAIN[args.emotion].format(question=question)
    qs = user_prompt.strip()

    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(device)

    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(device=device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(device=device, dtype=torch.float16)


    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True
        )

    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

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