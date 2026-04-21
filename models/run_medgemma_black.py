"""
Contrastive decoding variant: replaces dataset images with a plain black image.
Only runs yes_no=True + single conv_mode (save_scores config).
"""
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import argparse
from copy import deepcopy
import torch
import json
import os
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_utils import get_dataset
from emotion_prompts import SYSTEM_PROMPT, USER_PROMPTS


def build_messages(emotion: str):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {
                    "type": "text",
                    "text": USER_PROMPTS[emotion]
                    + " Question: {question}"
                    + " Please answer with 'Yes' or 'No'.",
                },
            ],
        },
    ]


def normalize_yes_no(text: str) -> str:
    t = text.strip().replace("\n", " ").lower()
    if t.startswith("yes"):
        return "Yes"
    if t.startswith("no"):
        return "No"
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_file", type=str,
                        default="output/phase_2/contrastive_decoding/medgemma_black_slake.jsonl")
    parser.add_argument("--emotion", type=str, default="default")
    parser.add_argument("--black_image", type=str, default="plain_black.png")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_id = "google/medgemma-1.5-4b-it"
    dtype = torch.bfloat16

    print(f"Loading model: {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    yes_id = processor.tokenizer.convert_tokens_to_ids("Yes")
    no_id = processor.tokenizer.convert_tokens_to_ids("No")

    # Load black image
    black_image = Image.open(args.black_image).convert("RGB")

    print(f"Loading dataset: {args.dataset}, split={args.split}, yes_no=True")
    samples = get_dataset(args.dataset, args.split, yes_no=True)
    print(f"Loaded {len(samples)} samples")

    messages_template = build_messages(args.emotion)
    dataset_lower = args.dataset.lower()

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in tqdm(samples, desc="Processing"):
            question = sample["question"]

            sample_messages = deepcopy(messages_template)
            # Always use black image instead of the real one
            sample_messages[-1]["content"][0]["image"] = black_image
            sample_messages[-1]["content"][1]["text"] = \
                sample_messages[-1]["content"][1]["text"].format(question=question)

            inputs = processor.apply_chat_template(
                sample_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=dtype)

            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            probs = generation["scores"][0][0].softmax(dim=-1)
            yes_raw = probs[yes_id].item()
            no_raw = probs[no_id].item()
            binary_total = yes_raw + no_raw
            if binary_total > 0:
                yes_prob = round(yes_raw / binary_total, 4)
                no_prob = round(no_raw / binary_total, 4)
            else:
                yes_prob = 0.0
                no_prob = 0.0

            input_len = inputs["input_ids"].shape[-1]
            decoded = processor.decode(
                generation["sequences"][0][input_len:], skip_special_tokens=True
            ).strip()
            decoded = normalize_yes_no(decoded)

            write_dict = {
                "question": question,
                "answer": sample["answer"],
                "model_answer": decoded,
                "yes_prob": yes_prob,
                "no_prob": no_prob,
                "max_prob": max(yes_prob, no_prob),
                "dataset": args.dataset,
                "conv_mode": "single",
                "emotion": args.emotion,
                "black_image": True,
            }
            f.write(json.dumps(write_dict, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()