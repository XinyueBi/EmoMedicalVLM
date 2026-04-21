#!/bin/bash
set -e

# ============================================================
# Contrastive Decoding: run yes_no=True, single conv, 
# but substitute ALL images with plain_black.png
# Covers: MedGemma + Lingshu x SLAKE + VQA-RAD x all emotions
# ============================================================

BLACK_IMG="plain_black.png"
OUT_DIR="output/phase_2/contrastive_decoding"
mkdir -p "$OUT_DIR"

EMOTIONS=(
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

DATASETS=("BoKelvin/SLAKE" "vqa-rad")
DATASET_TAGS=("slake" "vqarad")

# ----------------------------------------------------------
# MedGemma
# ----------------------------------------------------------
echo "====== MedGemma – Contrastive Decoding ======"

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    TAG="${DATASET_TAGS[$i]}"
    SPLIT="test"

    for EMOTION in "${EMOTIONS[@]}"; do
        # Skip default+multi (not applicable for black anyway, but consistent)
        if [ "$EMOTION" = "default" ]; then
            OUT_FILE="${OUT_DIR}/medgemma_black_${TAG}_default.jsonl"
        else
            OUT_FILE="${OUT_DIR}/medgemma_black_${TAG}_${EMOTION}.jsonl"
        fi

        if [ -f "$OUT_FILE" ]; then
            echo "Skip (exists): $OUT_FILE"
            continue
        fi

        echo "Running MedGemma | dataset=$TAG | emotion=$EMOTION"
        python models/run_medgemma_black.py \
            --dataset "$DATASET" \
            --split "$SPLIT" \
            --output_file "$OUT_FILE" \
            --emotion "$EMOTION" \
            --black_image "$BLACK_IMG"
    done
done

# ----------------------------------------------------------
# Lingshu
# ----------------------------------------------------------
echo "====== Lingshu – Contrastive Decoding ======"

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    TAG="${DATASET_TAGS[$i]}"
    SPLIT="test"

    for EMOTION in "${EMOTIONS[@]}"; do
        if [ "$EMOTION" = "default" ]; then
            OUT_FILE="${OUT_DIR}/lingshu_black_${TAG}_default.jsonl"
        else
            OUT_FILE="${OUT_DIR}/lingshu_black_${TAG}_${EMOTION}.jsonl"
        fi

        if [ -f "$OUT_FILE" ]; then
            echo "Skip (exists): $OUT_FILE"
            continue
        fi

        echo "Running Lingshu | dataset=$TAG | emotion=$EMOTION"
        python models/run_lingshu_black.py \
            --dataset "$DATASET" \
            --split "$SPLIT" \
            --output_file "$OUT_FILE" \
            --emotion "$EMOTION" \
            --black_image "$BLACK_IMG"
    done
done

echo "====== All contrastive decoding runs complete ======"
echo "Output saved to: $OUT_DIR"