#!/usr/bin/env python3
"""Pipeline YOLO + U-Net: detecta region de herida, recorta, segmenta.

Proposito: Hard Attention — YOLO localiza, U-Net segmenta solo la region relevante.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2, numpy as np, torch
from ultralytics import YOLO
from src.datasets.wound_dataset import get_default_transforms
from src.models.factory import create_model

DEVICE = "cuda"
PROJECT = Path(__file__).resolve().parent.parent.parent

# Cargar modelos
yolo = YOLO(str(PROJECT / "runs/detect/wound_detector/weights/best.pt"))
unet = create_model("unet", encoder_name="resnet50", pretrained=False)
ckpt = torch.load(str(PROJECT / "models/sensibilidad/unet_resnet50_v3.pth"), map_location=DEVICE, weights_only=False)
unet.load_state_dict(ckpt["model_state_dict"])
unet.to(DEVICE).eval()

def pipeline(image_path: str, output_dir: str = None):
    """YOLO detect -> Crop -> U-Net segment -> Stitch back."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: no se pudo leer {image_path}")
        return
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. YOLO: detectar bbox (conf mas bajo para capturar heridas finas)
    results = yolo(img, conf=0.15, verbose=False)
    boxes = results[0].boxes
    
    if boxes is None or len(boxes) == 0:
        print("YOLO: SIN DETECCION — pasando imagen completa a U-Net")
        bboxes = [(0, 0, w, h)]
    else:
        bboxes = []
        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box[:4].astype(int)
            # Expandir 20% para dar contexto a U-Net
            bw, bh = x2 - x1, y2 - y1
            pad_w, pad_h = int(bw * 0.2), int(bh * 0.2)
            x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
            x2, y2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
            bboxes.append((x1, y1, x2, y2))
        print(f"YOLO: {len(bboxes)} herida(s) detectada(s)")
    
    # 2. Por cada bbox: crop -> U-Net -> stitch
    full_mask = np.zeros((h, w), dtype=np.uint8)
    transform = get_default_transforms((384, 384))
    
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            continue
        
        crop = img_rgb[y1:y2, x1:x2]
        t = transform(image=crop)
        tensor = t["image"].unsqueeze(0).to(DEVICE)
        
        with torch.inference_mode():
            pred = torch.sigmoid(unet(tensor)).squeeze().cpu().numpy()
        
        mask_crop = (pred > 0.5).astype(np.uint8) * 255
        mask_crop = cv2.resize(mask_crop, (bw, bh))
        
        # Stitch back
        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], mask_crop)
    
    # 3. Guardar resultados
    total = h * w
    px = (full_mask > 0).sum()
    print(f"U-Net: {px}px ({px/total*100:.2f}%) de herida segmentada")
    
    out = Path(output_dir) if output_dir else Path(image_path).parent
    name = Path(image_path).stem
    
    cv2.imwrite(str(out / f"{name}_original.png"), img)
    cv2.imwrite(str(out / f"{name}_mascara_yolo_unet.png"), full_mask)
    
    overlay = img.copy()
    overlay[full_mask > 0] = [255, 200, 0]
    overlay = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    
    # Dibujar bboxes de YOLO en el overlay
    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    cv2.imwrite(str(out / f"{name}_superpuesto_yolo_unet.png"), overlay)
    
    print(f"Resultados: {name}_superpuesto_yolo_unet.png")
    print(f"  BBox YOLO: amarillo | Mascara U-Net: verde")
    
    return full_mask, bboxes

# ===================================================================
if __name__ == "__main__":
    for img in ["scripts/inference/miImage.jpg", "scripts/inference/imagen2.jpg"]:
        if Path(img).exists():
            print(f"\n{'='*60}")
            print(f"  {img}")
            print(f"{'='*60}")
            pipeline(img)
