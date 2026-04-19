import pandas as pd
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR


SLAKE_DIR = WORKSPACE_DIR / "SLAKE"
VQARAD_DIR = WORKSPACE_DIR / "vqa-rad"


def get_dataset(dataset_name, split, yes_no=False):
    dataset_name_lower = dataset_name.lower()

    if dataset_name in ['BoKelvin/SLAKE', 'SLAKE'] or dataset_name_lower == 'slake':
        assert split in ['train', 'validation', 'test'], \
            "Split must be one of 'train', 'validation', or 'test'."

        splits = {
            'train': 'train.json',
            'validation': 'validation.json',
            'test': 'test.json'
        }

        json_path = SLAKE_DIR / splits[split]
        imgs_dir = SLAKE_DIR / "imgs"

        if not json_path.exists():
            raise FileNotFoundError(f"SLAKE split file not found: {json_path}")
        if not imgs_dir.exists():
            raise FileNotFoundError(f"SLAKE image dir not found: {imgs_dir}")

        df = pd.read_json(json_path)
        df = df[df['q_lang'] == 'en']

        if yes_no:
            df = df[df['answer'].isin(['Yes', 'No'])]

        samples = []
        for _, row in df.iterrows():
            samples.append({
                "image": str(imgs_dir / row["img_name"]),
                "question": row["question"],
                "answer": row["answer"],
                "location": row["location"],
                "modality": row["modality"],
                "answer_type": row["answer_type"],
                "content_type": row["content_type"],
            })

        return samples

    elif dataset_name_lower == 'vqa-rad':
        assert split in ['train', 'test'], \
            "Split must be one of 'train' or 'test'."

        splits = {
            'train': VQARAD_DIR / 'data/train-00000-of-00001-eb8844602202be60.parquet',
            'test': VQARAD_DIR / 'data/test-00000-of-00001-e5bc3d208bb4deeb.parquet'
        }

        parquet_path = splits[split]
        if not parquet_path.exists():
            raise FileNotFoundError(f"VQA-RAD parquet file not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        df['answer'] = df['answer'].astype(str).str.strip()

        if yes_no:
            df = df[df['answer'].str.lower().isin(['yes', 'no'])]

        samples = []
        for _, row in df.iterrows():
            answer = row["answer"]
            if answer.lower() == "yes":
                answer = "Yes"
            elif answer.lower() == "no":
                answer = "No"

            samples.append({
                "image": row["image"],
                "question": row["question"],
                "answer": answer,
            })

        return samples
    
    elif dataset_name_lower == 'vindr_test':
        csv_path = "vindr_test/processed/annotations_test_processed.csv"

        df = pd.read_csv(csv_path)
        df = df[df['class_name'] != 'No finding']

        samples = []
        for _, row in df.iterrows():
            x_min = round(float(row["x_min"])/row["width"], 2)
            x_max = round(float(row["x_max"])/row["width"], 2)
            y_min = round(float(row["y_min"])/row["height"], 2)
            y_max = round(float(row["y_max"])/row["height"], 2)
            samples.append({
                "image": row["processed_path"],
                "class_name": row["class_name"],
                "bbox": [x_min, y_min, x_max, y_max],
            })

        return samples

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")