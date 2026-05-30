#!/usr/bin/env python3
"""Prueba watershed con FPN_EfficientNetB3 en miImage.jpg."""
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

DEVICE = "cuda"

# Cargar FPN_EfficientNetB3
ckpt = torch.load(
    str(PROJECT_ROOT / "models" / "screening" / "FPN_EfficientNetB3_best.pth"),
    map_location=DEVICE, weights_only=False
)
model = create_model("fpn", encoder_name="efficientnet-b3", pretrained=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.to(DEVICE).eval()

# Cargar imagen
img = cv2.imread(str(PROJECT_ROOT / "scripts" / "inference" / "miImage.jpg"))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Predecir
transform = get_default_transforms((384, 384))
tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
with torch.inference_mode():
    pred = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()

salida = PROJECT_ROOT / "scripts" / "inference" / "comparacion_screening"
salida.mkdir(parents=True, exist_ok=True)

# Probar varios umbrales
for umbral in [0.3, 0.4, 0.5]:
    mascara = (pred > umbral).astype(np.uint8) * 255
    mascara = cv2.resize(mascara, (img.shape[1], img.shape[0]))

    instance_map = watershed_instances(
        mascara, dist_threshold_ratio=0.3, min_instance_area=50, kernel_close=3
    )
    stats = compute_instance_stats(instance_map)

    pos_px = int((mascara > 0).sum())
    print(f"Umbral={umbral}: {pos_px}px positivos, {len(stats)} instancias")
    for s in stats:
        sid = s["instance_id"]
        apx = s["area_px"]
        bx, by, bw, bh = s["bbox_x"], s["bbox_y"], s["bbox_w"], s["bbox_h"]
        print(f"  [{sid}] area={apx}px bbox=({bx},{by},{bw}x{bh})")

    # Guardar overlay del mejor umbral
    if umbral == 0.4:
        # Instance overlay
        overlay = img.copy()
        colors = [(255, 200, 0), (0, 200, 255), (200, 0, 255)]
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

        overlay_img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        cv2.imwrite(str(salida / "miImage_watershed_instancias.png"), overlay_img)

        # CSV
        csv_path = salida / "miImage_watershed_instancias.csv"
        with open(csv_path, "w", newline="") as f:
            fields = ["instance_id", "area_px", "area_pct", "bbox_x", "bbox_y",
                      "bbox_w", "bbox_h", "centroid_x", "centroid_y"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for s in stats:
                w.writerow(s)
        print(f"\nGuardado: {overlay_img.shape}")
        print(f"CSV: {csv_path}")
