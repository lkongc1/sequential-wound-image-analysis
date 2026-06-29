#!/usr/bin/env python3
"""Test de instancias con FPN_EfficientNetB3."""
import sys
from pathlib import Path
import csv

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import create_model
from src.datasets.wound_dataset import get_default_transforms
from src.inference.postprocessing import watershed_instances, compute_instance_stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# Cargar modelo FPN_EfficientNetB3
ckpt_path = PROJECT_ROOT / "models" / "screening" / "FPN_EfficientNetB3_best.pth"
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model = create_model("fpn", encoder_name="efficientnet-b3", pretrained=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.to(DEVICE).eval()
print("Modelo FPN_EfficientNetB3 cargado")

# Cargar imagen
img_path = PROJECT_ROOT / "scripts" / "inference" / "imagen2.jpg"
original = cv2.imread(str(img_path))
img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# Predecir
transform = get_default_transforms((384, 384))
tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
with torch.inference_mode():
    pred = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()

mascara = (pred > 0.5).astype(np.uint8) * 255
mascara = cv2.resize(mascara, (original.shape[1], original.shape[0]))

# Watershed instances
instance_map = watershed_instances(mascara, dist_threshold_ratio=0.3, min_instance_area=50, kernel_close=3)
stats = compute_instance_stats(instance_map)

print(f"\nMascara: {int((mascara>0).sum())} px positivos")
print(f"Instancias encontradas: {len(stats)}")
for s in stats:
    sid = s["instance_id"]
    apx = s["area_px"]
    pct = s["area_pct"]
    bx, by, bw, bh = s["bbox_x"], s["bbox_y"], s["bbox_w"], s["bbox_h"]
    cx, cy = s["centroid_x"], s["centroid_y"]
    print(f"  Instancia {sid}: area={apx}px ({pct:.2f}%), bbox=({bx},{by},{bw}x{bh}), centroid=({cx:.0f},{cy:.0f})")

# Guardar resultados
salida = PROJECT_ROOT / "scripts" / "inference" / "comparacion_screening"
salida.mkdir(parents=True, exist_ok=True)

# Instance overlay
overlay = original.copy()
colors = [(255, 200, 0), (0, 200, 255), (200, 0, 255), (0, 100, 255), (255, 0, 100)]
for s in stats:
    sid = s["instance_id"]
    color = colors[(sid - 1) % len(colors)]
    mask_i = (instance_map == sid).astype(np.uint8) * 255
    overlay[mask_i > 0] = color
    contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    if contours:
        M = cv2.moments(contours[0])
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(overlay, str(sid), (cx - 5, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

overlay_img = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)
out_png = salida / "FPN_EfficientNetB3_instancias.png"
cv2.imwrite(str(out_png), overlay_img)

# CSV
csv_path = salida / "FPN_EfficientNetB3_instancias.csv"
with open(csv_path, "w", newline="") as f:
    fields = ["instance_id", "area_px", "area_pct", "bbox_x", "bbox_y",
              "bbox_w", "bbox_h", "centroid_x", "centroid_y"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for s in stats:
        w.writerow(s)

print(f"\nGuardado: {out_png}")
print(f"Guardado: {csv_path}")
print("\nOK - Instancias funcionando correctamente")
