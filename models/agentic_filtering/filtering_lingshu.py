from PIL import Image
import argparse
import json
import os
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data_utils import get_dataset
from emotion_prompts import SYSTEM_PROMPT, USER_PROMPTS


def get_conversation_prompt(emotion, conv_mode, question):
    user_prompt = USER_PROMPTS[emotion]
    if conv_mode == "single":
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": "image_path"},
                {"type": "text", "text": user_prompt + f" {question}"},
            ]},
        ]
    else:
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": "I'm here to help."}]},
            {"role": "user", "content": [
                {"type": "image", "image": "image_path"},
                {"type": "text", "text": f"{question}"},
            ]},
        ]


EXTRACTION_SYSTEM = """You are an extraction module.

            Your task:
            Read the full conversation history and extract ONLY the relevant question or request to answer. 
            Ignore all irrelvant information.

            Output rules:
            - Return only the extracted question or request.
            - Do not answer it.
            - Do not explain anything.
            - Do not add extra text.

            If there is no clear question or request, return exactly:
            NO_QUESTION"""

EXTRACTION_USER_TEMPLATE = """Conversation history:
            {conversation_history}

            Extract:"""


def build_text_only_prompt(conversation_history_str):
    """Plain text prompt, no image tokens — extraction is text-only."""
    user = EXTRACTION_USER_TEMPLATE.format(conversation_history=conversation_history_str)
    prompt = (
        f"<|im_start|>system\n{EXTRACTION_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return {
        "prompt": prompt,
        "multi_modal_data": {},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str,
                        default="output/phase_2/agentic_filtering/lingshu.jsonl")
    parser.add_argument("--model_id", type=str,
                        default="lingshu-medical-mllm/Lingshu-I-8B")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from vllm import LLM, SamplingParams

    output_dir = Path(args.output_file).parent
    os.makedirs(output_dir, exist_ok=True)

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
        repetition_penalty=1.3,
        max_tokens=128,
        stop=["<|im_end|>"],
    )

    emotion_list = [
        "direct_patient_neutral", "direct_patient_fear_anxiety",
        "direct_patient_anger_frustration", "direct_patient_sadness_distress",
        "direct_clinician_neutral", "direct_clinician_fear_anxiety",
        "direct_clinician_anger_frustration", "direct_clinician_sadness_distress",
        "indirect_patient_neutral", "indirect_patient_fear_anxiety",
        "indirect_patient_anger_frustration", "indirect_patient_sadness_distress",
        "indirect_clinician_neutral", "indirect_clinician_fear_anxiety",
        "indirect_clinician_anger_frustration", "indirect_clinician_sadness_distress",
    ]
    conv_modes = ["single", "multi"]

    samples = get_dataset("BoKelvin/SLAKE", "test", False)[:50]

    with open(args.output_file, "w", encoding="utf-8") as f:
        for emotion in emotion_list:
            for conv_mode in conv_modes:
                print(f"Processing emotion: {emotion}, conversation mode: {conv_mode}")
                for sample in tqdm(samples, desc=f"Processing {emotion} - {conv_mode}"):
                    question = sample["question"]
                    messages = get_conversation_prompt(emotion, conv_mode, question)
                    conversation_history = json.dumps(messages, ensure_ascii=False)
                    llm_inputs = build_text_only_prompt(conversation_history)

                    try:
                        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
                        decoded = outputs[0].outputs[0].text.strip()
                    except Exception as e:
                        decoded = f"ERROR: {e}"

                    write_dict = {
                        **sample,
                        "emotion": emotion,
                        "conv_mode": conv_mode,
                        "model_answer": decoded,
                        "correct": (decoded == question),
                    }
                    f.write(json.dumps(write_dict, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()