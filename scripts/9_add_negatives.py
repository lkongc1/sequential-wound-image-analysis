#!/usr/bin/env python3
"""Agrega imagenes negativas (piel sana) de AZH BG al dataset + crea mascaras vacias."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2, numpy as np, pandas as pd

PROJECT = Path(__file__).parent.parent
BG_DIR = PROJECT / "data" / "raw" / "AZH_Wound" / "Train" / "BG"
OUT_DIR = PROJECT / "data" / "processed" / "train"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROJECT / "data" / "processed" / "dataset_final.csv")
print(f"Dataset actual: {len(df)} filas")

new_rows = []
for img_path in sorted(BG_DIR.glob("*.jpg")):
    # Leer imagen original
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]
    
    # Guardar en processed/train
    out_name = f"neg_azh_{img_path.stem}.png"
    out_img = OUT_DIR / out_name
    cv2.imwrite(str(out_img), img)
    
    # Crear mascara vacia (todo negro = sin herida)
    mask = np.zeros((h, w), dtype=np.uint8)
    out_mask = OUT_DIR / f"neg_azh_{img_path.stem}_mask.png"
    cv2.imwrite(str(out_mask), mask)
    
    new_rows.append({
        "filename": out_name,
        "source": "azh_bg",
        "split": "train",
        "image_path": str(out_img),
        "mask_path": str(out_mask),
        "wound_percentage": 0.0,
        "brightness_mean": img.mean(),
        "brightness_std": img.std(),
        "contrast_rms": 0.0,
        "mask_area_pixels": 0,
        "mask_edge_density": 0.0,
        "is_empty": True,
        "is_outlier": False,
        "outlier_reason": "",
        "review_status": "auto",
    })

new_df = pd.DataFrame(new_rows)
df = pd.concat([df, new_df], ignore_index=True)
df.to_csv(PROJECT / "data" / "processed" / "dataset_final.csv", index=False)

print(f"Agregadas {len(new_rows)} imagenes negativas (AZH BG)")
print(f"Dataset final: {len(df)} filas (train={len(df[df['split']=='train'])}, test={len(df[df['split']=='test'])})")
print(f"Negativos totales: {df['is_empty'].sum()}")
