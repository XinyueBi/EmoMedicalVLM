"""
Contrastive decoding variant for Lingshu: replaces dataset images with a plain black image.
Only runs yes_no=True + single conv_mode (save_scores config).
"""
import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
import sys
import dataclasses
from enum import IntEnum, auto
from typing import List, Tuple, Union

from PIL import Image
from tqdm import tqdm

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
    return "Unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_file", type=str,
                        default="output/phase_2/contrastive_decoding/lingshu_black_slake.jsonl")
    parser.add_argument("--emotion", type=str, default="default")
    parser.add_argument("--black_image", type=str, default="plain_black.png")
    parser.add_argument("--model_id", type=str, default="lingshu-medical-mllm/Lingshu-I-8B")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Import vLLM inside main() to avoid multiprocessing fork issues
    from vllm import LLM, SamplingParams

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
        repetition_penalty=1.3,      # prevent degeneration with black image
        max_tokens=16,               # Yes/No only needs a few tokens
        stop=["<|im_end|>", "\n"],   # stop at turn boundary
    )

    samples = get_dataset(args.dataset, args.split, yes_no=True)
    print(f"Loaded {len(samples)} samples")

    black_image = Image.open(args.black_image).convert("RGB")

    messages_template = [
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
                            + " Please answer with 'Yes' or 'No'.",
                },
            ],
        },
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in tqdm(samples, desc="Processing"):
            try:
                question = sample["question"]

                sample_messages = deepcopy(messages_template)
                sample_messages[-1]["content"][0]["image"] = black_image
                sample_messages[-1]["content"][1]["text"] = \
                    sample_messages[-1]["content"][1]["text"].format(question=question)

                llm_inputs = process_messages(sample_messages)
                outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
                decoded = normalize_yes_no(outputs[0].outputs[0].text.strip())

                write_dict = {
                    "question": question,
                    "answer": sample["answer"],
                    "model_answer": decoded,
                    "dataset": args.dataset,
                    "conv_mode": "single",
                    "emotion": args.emotion,
                    "black_image": True,
                }
            except Exception as e:
                write_dict = {
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "model_answer": "Unknown",
                    "dataset": args.dataset,
                    "conv_mode": "single",
                    "emotion": args.emotion,
                    "black_image": True,
                    "error": str(e),
                }

            f.write(json.dumps(write_dict, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()