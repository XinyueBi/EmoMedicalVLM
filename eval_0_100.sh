#!/bin/bash
set -e

# ===== configuration =====
OUTPUT_DIR="/workspace/EmoMedicalVLM/output/phase_2/Lingshu"
FILE_PREFIX="lingshu_"
BATCH_SIZE=4
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

echo "======================================"
echo " Running 0-100 Evaluation"
echo " Output dir: $OUTPUT_DIR"
echo "======================================"

python eval_0_100.py \
  --output_dir $OUTPUT_DIR \
  --file_prefix $FILE_PREFIX \
  --batch_size $BATCH_SIZE \
  --model_name $MODEL_NAME

echo "======================================"
echo " Done!"
echo " Results saved as eval_0_100_*.jsonl"
echo "======================================"