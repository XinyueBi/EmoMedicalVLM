import argparse
import torch
import json
import os
from tqdm import tqdm
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data_utils import get_dataset
from emotion_prompts import USER_PROMPTS

from transformers import AutoModelForCausalLM, AutoProcessor


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--output_file", type=str, default="output/Hulu-Med/hulumed.jsonl")
parser.add_argument("--emotion", type=str, default="default")
parser.add_argument("--model_path", type=str, default="ZJU-AI4H/Hulu-Med-4B")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--yes_no", action="store_true", help="Whether to filter yes/no questions and force a Yes/No answer.")
parser.add_argument("--conv_mode", type=str, default="single", choices=["single", "multi"], help="Whether to use single-turn or multi-turn conversation format.")
args = parser.parse_args()

BLACK_IMAGE_PATH = Path(__file__).resolve().parent.parent.parent / "plain_black.png"
black_image = Image.open(BLACK_IMAGE_PATH).convert("RGB")

if args.emotion == "default" and args.conv_mode == "multi":
    exit(0)

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

save_scores = args.yes_no and args.conv_mode == "single"
if save_scores:
    yes_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize("Yes"))[0]
    no_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize("No"))[0]

for sample in tqdm(samples, desc="Processing samples"):
    image_content = black_image.copy()

    question = sample["question"]

    if args.conv_mode == "single":
        prompt_text = (
            USER_PROMPTS[args.emotion]
            + " Question: {question}"
            + (" Please answer with 'Yes' or 'No'." if args.yes_no else "")
        ).format(question=question)

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_content,
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            }
        ]
    else:
        second_user_text = (
            " Question: {question}"
            + (" Please answer with 'Yes' or 'No'." if args.yes_no else "")
        ).format(question=question)

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": USER_PROMPTS[args.emotion],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "I'm here to help.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_content,
                    },
                    {
                        "type": "text",
                        "text": second_user_text,
                    },
                ],
            },
        ]

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
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

    if save_scores:
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                **gen_kwargs,
                output_scores=True,
                return_dict_in_generate=True,
            )

        probs = generation.scores[0][0].softmax(dim=-1)

        token_max_prob, _ = probs.max(dim=-1)
        token_max_prob = round(token_max_prob.item(), 4)

        yes_raw = probs[yes_id].item()
        no_raw = probs[no_id].item()

        binary_total = yes_raw + no_raw
        if binary_total > 0:
            yes_prob = round(yes_raw / binary_total, 4)
            no_prob = round(no_raw / binary_total, 4)
        else:
            yes_prob = 0.0
            no_prob = 0.0

        max_prob = max(yes_prob, no_prob)

        decoded = processor.batch_decode(
            generation.sequences,
            skip_special_tokens=True,
            use_think=False,
        )[0].strip()

        decoded_lower = decoded.lower().strip()

        if decoded_lower.startswith("answer:"):
            decoded_lower = decoded_lower.replace("answer:", "", 1).strip()

        if decoded_lower.startswith("yes"):
            decoded = "Yes"
        elif decoded_lower.startswith("no"):
            decoded = "No"

    else:
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)

        decoded = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            use_think=False,
        )[0].strip()

        if args.yes_no:
            decoded_lower = decoded.lower().strip()

            if decoded_lower.startswith("answer:"):
                decoded_lower = decoded_lower.replace("answer:", "", 1).strip()

            if decoded_lower.startswith("yes"):
                decoded = "Yes"
            elif decoded_lower.startswith("no"):
                decoded = "No"

    write_dict = {
        "question": question,
        "answer": sample["answer"],
        "answer_type": "CLOSED" if args.yes_no else "OPEN",
        "model_answer": decoded,
        "dataset": args.dataset,
        "conv_mode": args.conv_mode,
    }

    if save_scores:
        write_dict["max_prob"] = max_prob
        write_dict["yes_prob"] = yes_prob
        write_dict["no_prob"] = no_prob

    with open(args.output_file, "a") as f:
        f.write(json.dumps(write_dict) + "\n")