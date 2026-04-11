import argparse
import json
import os
from io import BytesIO
from copy import deepcopy
from pathlib import Path
import sys
import dataclasses
from enum import IntEnum, auto
from typing import List, Tuple, Union

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_utils import get_dataset
from emotion_prompts import SYSTEM_PROMPT, USER_PROMPTS


class SeparatorStyle(IntEnum):
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    name: str
    system_template: str = "{system_message}"
    system_message: str = ""
    roles: Tuple[str, str] = ("USER", "ASSISTANT")
    messages: List[List[str]] = dataclasses.field(default_factory=list)
    offset: int = 0
    sep_style: SeparatorStyle = SeparatorStyle.MPT
    sep: str = "\n"
    sep2: str = None
    stop_str: Union[str, List[str]] = None
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        system_prompt = self.system_template.format(system_message=self.system_message)
        ret = system_prompt + self.sep
        for role, message in self.messages:
            if message:
                ret += role + message + self.sep
            else:
                ret += role
        return ret


def process_messages(messages):
    conv = Conversation(
        name="internvl3",
        system_template="<|im_start|>system\n{system_message}",
        system_message="",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    )

    imgs = []

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "system":
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        text += item["text"]
            conv.system_message = text
            continue

        prompt_role = conv.roles[0] if role == "user" else conv.roles[1]

        if isinstance(content, str):
            conv.messages.append([prompt_role, content])
        elif isinstance(content, list):
            text = ""
            for item in content:
                if item["type"] == "text":
                    text += item["text"]
                elif item["type"] == "image":
                    text += "\n<IMG_CONTEXT>"
                    image = item["image"]
                    if isinstance(image, str):
                        image = Image.open(image).convert("RGB")
                    imgs.append(image)
            conv.messages.append([prompt_role, text])

    conv.messages.append([conv.roles[1], None])

    prompt = conv.get_prompt()
    mm_data = {}
    if len(imgs) > 0:
        mm_data["image"] = imgs

    return {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }


def normalize_yes_no(text: str) -> str:
    t = text.strip().lower()
    if t.startswith("yes"):
        return "Yes"
    if t.startswith("no"):
        return "No"
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_file", type=str, default="output/lingshu.jsonl")
    parser.add_argument("--emotion", type=str, default="default")
    parser.add_argument("--yes_no", action="store_true")
    parser.add_argument("--conv_mode", type=str, default="single", choices=["single", "multi"])
    parser.add_argument("--model_id", type=str, default="lingshu-medical-mllm/Lingshu-I-8B")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_num_seqs", type=int, default=1)
    args = parser.parse_args()

    if args.emotion == "default" and args.conv_mode == "multi":
        print("Skipping default + multi")
        raise SystemExit(0)

    output_dir = Path(args.output_file).parent
    os.makedirs(output_dir, exist_ok=True)

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    llm = LLM(
        model=args.model_id,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192,
        max_num_seqs=1,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        max_tokens=256,
    )

    samples = get_dataset(args.dataset, args.split, args.yes_no)

    if args.conv_mode == "single":
        messages = [
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
                        "text": USER_PROMPTS[args.emotion]
                                + " Question: {question}"
                                + (" Please answer with 'Yes' or 'No'." if args.yes_no else "")
                    },
                ],
            },
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": USER_PROMPTS[args.emotion]}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Understood."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": None},
                    {
                        "type": "text",
                        "text": " Question: {question}"
                                + (" Please answer with 'Yes' or 'No'." if args.yes_no else "")
                    },
                ],
            },
        ]

    for sample in tqdm(samples, desc="Processing samples"):
        try:
            if "SLAKE" in args.dataset:
                image = Image.open(sample["image"]).convert("RGB")
            elif "vqa-rad" in args.dataset.lower():
                image = Image.open(BytesIO(sample["image"]["bytes"])).convert("RGB")
            else:
                raise ValueError(f"Unsupported dataset format: {args.dataset}")

            question = sample["question"]

            sample_messages = deepcopy(messages)
            sample_messages[-1]["content"][0]["image"] = image
            sample_messages[-1]["content"][1]["text"] = sample_messages[-1]["content"][1]["text"].format(
                question=question
            )

            llm_inputs = process_messages(sample_messages)
            outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
            decoded = outputs[0].outputs[0].text.strip()

            if args.yes_no:
                decoded = normalize_yes_no(decoded)

            write_dict = {
                "question": sample["question"],
                "answer": sample["answer"],
                "model_answer": decoded,
                "dataset": args.dataset,
                "conv_mode": args.conv_mode,
            }

        except Exception as e:
            write_dict = {
                "question": sample["question"],
                "answer": sample["answer"],
                "model_answer": "",
                "dataset": args.dataset,
                "conv_mode": args.conv_mode,
                "model_name": args.model_id,
                "error": str(e),
            }

        with open(args.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(write_dict, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()