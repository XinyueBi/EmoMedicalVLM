#!/bin/bash
set -u

MODEL_NAME="lingshu-medical-mllm/Lingshu-I-8B"
SCRIPT_PATH="models/run_lingshu.py"
OUTPUT_DIR="output/phase_2/Lingshu"
LOG_DIR="logs/lingshu"

GPU_ID=0
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=8192

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

emotions=(
  "default"
  "direct_patient_neutral"
  "direct_patient_fear_anxiety"
  "direct_patient_anger_frustration"
  "direct_patient_sadness_distress"
  "direct_clinician_neutral"
  "direct_clinician_fear_anxiety"
  "direct_clinician_anger_frustration"
  "direct_clinician_sadness_distress"
  "indirect_patient_neutral"
  "indirect_patient_fear_anxiety"
  "indirect_patient_anger_frustration"
  "indirect_patient_sadness_distress"
  "indirect_clinician_neutral"
  "indirect_clinician_fear_anxiety"
  "indirect_clinician_anger_frustration"
  "indirect_clinician_sadness_distress"
)

datasets=("SLAKE" "vqa-rad")
conv_modes=("single" "multi")

run_job () {
  local dataset="$1"
  local conv_mode="$2"
  local emotion="$3"
  local yesno_flag="$4"

  local suffix=""
  local yesno_arg=""
  local job_type="open"

  if [ "$yesno_flag" = "closed" ]; then
    suffix="_closed"
    yesno_arg="--yes_no"
    job_type="closed"
  fi

  local output_file="${OUTPUT_DIR}/lingshu_${dataset}_${conv_mode}_${emotion}${suffix}.jsonl"
  local log_file="${LOG_DIR}/lingshu_${dataset}_${conv_mode}_${emotion}${suffix}.log"

  if [ -f "$output_file" ] && [ -s "$output_file" ]; then
    echo "[SKIP] Exists: $output_file"
    return 0
  fi

  echo "[RUN ] dataset=$dataset conv_mode=$conv_mode emotion=$emotion type=$job_type"
  echo "[LOG ] $log_file"

  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$SCRIPT_PATH" \
    --model_id "$MODEL_NAME" \
    --dataset "$dataset" \
    --split "test" \
    --emotion "$emotion" \
    --conv_mode "$conv_mode" \
    --output_file "$output_file" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    $yesno_arg > "$log_file" 2>&1

  local exit_code=$?

  if [ $exit_code -ne 0 ]; then
    echo "[FAIL] dataset=$dataset conv_mode=$conv_mode emotion=$emotion type=$job_type"
    echo "[FAIL] See log: $log_file"
    return $exit_code
  fi

  if [ ! -f "$output_file" ] || [ ! -s "$output_file" ]; then
    echo "[WARN] Finished but output file missing or empty: $output_file"
  else
    echo "[DONE] $output_file"
  fi

  return 0
}

for emotion in "${emotions[@]}"; do
  for dataset in "${datasets[@]}"; do
    for conv_mode in "${conv_modes[@]}"; do

      if [ "$emotion" = "default" ] && [ "$conv_mode" = "multi" ]; then
        echo "[SKIP] default + multi is disabled"
        continue
      fi

      run_job "$dataset" "$conv_mode" "$emotion" "open"
      run_job "$dataset" "$conv_mode" "$emotion" "closed"

    done
  done
done

echo "All jobs processed."