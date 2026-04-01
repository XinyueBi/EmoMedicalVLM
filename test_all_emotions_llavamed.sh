#!/bin/bash

set -euo pipefail

mkdir -p output/LLaVA-Med

for emotion in \
  "default" \
  "main_patient_neutral" \
  "main_patient_fear_anxiety" \
  "main_patient_anger_frustration" \
  "main_patient_sadness_distress" \
  "main_clinician_neutral" \
  "main_clinician_fear_anxiety" \
  "main_clinician_anger_frustration" \
  "main_clinician_sadness_distress"
do
    echo "Testing emotion: $emotion"
    python models/run_llavamed.py \
      --dataset "SLAKE" \
      --split "test" \
      --emotion "$emotion" \
      --output_file "output/LLaVA-Med/llavamed_${emotion}.jsonl"
done