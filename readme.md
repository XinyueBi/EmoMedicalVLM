## Download and extract the SLAKE dataset

```bash
apt install -y git-lfs
git clone https://huggingface.co/datasets/BoKelvin/SLAKE
cd SLAKE
git lfs pull
unzip imgs.zip
```

## Download the VQA dataset

```bash
apt install -y git-lfs
git clone https://huggingface.co/datasets/flaviagiammarino/vqa-rad
cd vqa-rad
git lfs pull
```

## Download the VinDr dataset, test split from kaggle and annotations from physionet (Khoi only)

[Kaggle](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/data)

[PhysioNet](https://physionet.org/content/vindr-cxr/1.0.0/annotations/annotations_test.csv)

Then put both into `vindr_test` before running:

```bash
pip install pydicom
python process_vindr.py
```

## Test runs with MedGemma

For a single run with default prompt, run 

```python models/run_medgemma.py``` 

or completed run with all emotions with ```bash test_all_emotions.sh```

## Warning

Due to `transformers` and its models being version-sensitive, all dependencies in `requirements.txt` are for reference only.