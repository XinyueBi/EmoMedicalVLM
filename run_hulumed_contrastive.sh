#!/bin/bash

for emotion in "default" "direct_patient_neutral" "direct_patient_fear_anxiety" \
        "direct_patient_anger_frustration" "direct_patient_sadness_distress" \
        "direct_clinician_neutral" "direct_clinician_fear_anxiety" \
        "direct_clinician_anger_frustration" "direct_clinician_sadness_distress" \
        "indirect_patient_neutral" "indirect_patient_fear_anxiety" \
        "indirect_patient_anger_frustration" "indirect_patient_sadness_distress" \
        "indirect_clinician_neutral" "indirect_clinician_fear_anxiety" \
        "indirect_clinician_anger_frustration" "indirect_clinician_sadness_distress" ; do
    for dataset in "SLAKE" "vqa-rad" ; do
        echo "Testing emotion: $emotion, dataset: $dataset, conv_mode: single, yes/no answer, black image"

        python models/contrastive_decoding/run_hulumed_contrastive.py \
            --dataset "$dataset" \
            --split "test" \
            --emotion "$emotion" \
            --conv_mode "single" \
            --yes_no \
            --output_file "output/phase_2/contrastive_decoding/hulumed_${dataset}_single_${emotion}_closed.jsonl"
    done
done