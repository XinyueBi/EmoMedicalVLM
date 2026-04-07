import argparse
import torch
import json
import os
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import get_dataset
from emotion_prompts import SYSTEM_PROMPT, INDUCED_STORY

from transformers import AutoModelForCausalLM, AutoProcessor


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--output_file", type=str, default="output/Hulu-Med/hulumed_induced.jsonl")
parser.add_argument("--emotion", type=str, required=True)
parser.add_argument("--model_path", type=str, default="ZJU-AI4H/Hulu-Med-4B")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--yes_no", action="store_true", help="Whether to filter yes/no questions and force a Yes/No answer.")
args = parser.parse_args()

output_dir = Path(args.output_file).parent
os.makedirs(output_dir, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)

processor = AutoProcessor.from_pretrained(
    args.model_path,
    trust_remote_code=True,
)

samples = get_dataset(args.dataset, args.split, args.yes_no)
if args.max_samples is not None:
    samples = samples[:args.max_samples]

for sample in tqdm(samples, desc="Processing samples"):
    question = sample["question"]

    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": INDUCED_STORY[args.emotion],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I'm here to help.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": {
                        "image_path": sample["image"],
                    },
                },
                {
                    "type": "text",
                    "text": f"Question: {question}" + (" Please answer with 'Yes' or 'No'." if args.yes_no else ""),
                },
            ],
        },
    ]

    inputs = processor(
        conversation=conversation,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
    }

    if args.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            gen_kwargs["top_p"] = args.top_p
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    decoded = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        use_think=False,
    )[0].strip()

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