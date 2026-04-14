import argparse
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_score(response: str):
    """
    Extract score from model response.
    Expected format: 'Score: <0-100>'
    """
    if not response:
        return None

    match = re.search(r"Score:\s*(\d+)", response)
    if match:
        score = int(match.group(1))
        return max(0, min(100, score))

    numbers = re.findall(r"\d+", response)
    if numbers:
        score = int(numbers[0])
        return max(0, min(100, score))

    return None


def score_to_label(score):
    """
    Optional coarse label derived from numeric score.
    """
    if score is None:
        return "Unknown"
    if score == 100:
        return "Correct"
    if score >= 60:
        return "Partially Correct"
    if score >= 1:
        return "Incorrect"
    return "Irrelevant"


def build_prompt(question, ground_truth, llm_response):
    system_prompt = (
        "You are an expert evaluator for a medical visual question answering system.\n"
        "You will be given a question, a ground truth answer, and a model response.\n"
        "Your task is to assign a score from 0 to 100 indicating how correct the model response is.\n\n"
        "Scoring guideline:\n"
        "- 100: perfectly correct, fully matches the ground truth answer\n"
        "- 70-99: mostly correct, with only minor issues or omissions\n"
        "- 40-69: partially correct, with significant missing details or some inaccuracies\n"
        "- 1-39: mostly incorrect, but contains a small amount of relevant or correct information\n"
        "- 0: completely incorrect or irrelevant\n\n"
        "For Yes/No questions:\n"
        "- Score 100 if the answer is correct\n"
        "- Score 0 if the answer is incorrect\n\n"
        "Be strict and consistent.\n"
        "Output ONLY in the following format:\n"
        "Score: <number between 0 and 100>\n"
        "Do not output anything else."
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Ground Truth Answer: {ground_truth}\n"
        f"LLM Response: {llm_response}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory containing jsonl prediction files.",
    )
    parser.add_argument(
        "--file_prefix",
        type=str,
        default="medgemma_",
        help="Only evaluate files whose names start with this prefix.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for judge model evaluation.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Judge model name.",
    )
    args = parser.parse_args()

    print(f"Loading judge model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    all_files = sorted(os.listdir(args.output_dir))
    target_files = [
        f for f in all_files
        if f.startswith(args.file_prefix) and f.endswith(".jsonl")
    ]

    if not target_files:
        print(f"No matching files found in {args.output_dir} with prefix '{args.file_prefix}'")
        return

    for file_name in target_files:
        input_path = os.path.join(args.output_dir, file_name)
        output_path = os.path.join(args.output_dir, "eval_0_100_" + file_name)

        print(f"\nEvaluating {file_name} ...")

        with open(input_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        for start in tqdm(range(0, len(data), args.batch_size), desc=file_name):
            batch = data[start:start + args.batch_size]

            texts = []
            for item in batch:
                question = item.get("question", "")
                ground_truth = item.get("answer", "")
                llm_response = item.get("model_answer", "")

                input_messages = build_prompt(
                    question=question,
                    ground_truth=ground_truth,
                    llm_response=llm_response,
                )

                text = tokenizer.apply_chat_template(
                    input_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                texts.append(text)

            model_inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for item, response in zip(batch, responses):
                raw_response = response.strip()
                score = parse_score(raw_response)

                item["score"] = score
                item["evaluation"] = score_to_label(score)
                item["raw_judge_response"] = raw_response

        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()