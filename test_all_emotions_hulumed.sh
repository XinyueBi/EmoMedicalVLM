#!/bin/bash

set -euo pipefail

mkdir -p output/Hulu-Med/closed

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
    echo "Testing closed yes/no emotion: $emotion"
    python models/run_hulumed.py \
      --dataset "SLAKE" \
      --split "test" \
      --emotion "$emotion" \
      --yes_no \
      --output_file "output/Hulu-Med/closed/hulumed_${emotion}.jsonl"
done