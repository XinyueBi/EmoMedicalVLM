import os
from PIL import Image
import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut
import pandas as pd
from tqdm import tqdm

input_dir = "vindr_test"
output_dir = os.path.join(input_dir, "processed")
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("vindr_test/annotations_test.csv")

def load_cxr_dicom(path, use_voi=True, to_uint8=True):
    ds = pydicom.dcmread(path)

    # 1) raw pixel -> ndarray
    img = ds.pixel_array.astype(np.float32)

    # 2) modality LUT / rescale first
    img = apply_modality_lut(img, ds).astype(np.float32)

    # 3) VOI LUT / windowing if wanted
    if use_voi:
        try:
            img = apply_voi_lut(img, ds).astype(np.float32)
        except Exception:
            pass

    # 4) fix MONOCHROME1
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img

    # 5) normalize
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()

    if to_uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    return img

size_map = {}
path_map = {}

for img_name in tqdm(os.listdir(input_dir)):
    if not img_name.endswith(".dicom"):
        continue

    img_path = os.path.join(input_dir, img_name)
    img = load_cxr_dicom(img_path)

    height, width = img.shape[:2]

    out_name = img_name.replace(".dicom", ".png")   # đổi thành .jpg nếu muốn
    out_path = os.path.join(output_dir, out_name)

    Image.fromarray(img).save(out_path)

    image_id = img_name.replace(".dicom", "")
    size_map[image_id] = (width, height)
    path_map[image_id] = out_path


df["width"] = df["image_id"].map(lambda x: size_map.get(x, (None, None))[0])
df["height"] = df["image_id"].map(lambda x: size_map.get(x, (None, None))[1])
df["processed_path"] = df["image_id"].map(path_map)

df.to_csv(os.path.join(output_dir, "annotations_test_processed.csv"), index=False)