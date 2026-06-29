#!/usr/bin/env python3
"""Compara los 12 modelos del screening sobre una imagen.

Para cada modelo, predice la máscara y guarda:
  - {modelo}_superpuesto.png (overlay visual)
  - Un CSV con todas las métricas calculadas

Uso:
    python scripts/inference/compare_models.py [--imagen ruta] [--salida dir]
"""

import sys
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import create_model
from src.datasets.wound_dataset import get_default_transforms

SCREENING_DIR = PROJECT_ROOT / "models" / "screening"
DEFAULT_IMAGEN = PROJECT_ROOT / "scripts" / "inference" / "imagen2.jpg"
SALIDA_DIR = PROJECT_ROOT / "scripts" / "inference" / "comparacion_screening"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (384, 384)
TRANSFORM = get_default_transforms(IMAGE_SIZE)


MODELOS = [
    ("DeepLabV3_ResNeXt50", "deeplabv3", "resnext50_32x4d"),
    ("DeepLabV3Plus_ResNeXt50", "deeplabv3plus", "resnext50_32x4d"),
    ("UNet_ResNeXt50", "unet", "resnext50_32x4d"),
    ("FPN_ResNeXt50", "fpn", "resnext50_32x4d"),
    ("DeepLabV3Plus_ResNet101", "deeplabv3plus", "resnet101"),
    ("UNet_ResNet101", "unet", "resnet101"),
    ("FPN_ResNet101", "fpn", "resnet101"),
    ("DeepLabV3Plus_EfficientNetB3", "deeplabv3plus", "efficientnet-b3"),
    ("UNet_EfficientNetB3", "unet", "efficientnet-b3"),
    ("FPN_EfficientNetB3", "fpn", "efficientnet-b3"),
    ("UNet_SegFormer", "unet", "mit_b2"),
    ("FPN_SegFormer", "fpn", "mit_b2"),
]

COLOR_OVERLAY = (255, 200, 0)  # BGR verde azulado


def cargar_modelo(nombre_arch, arch, encoder):
    """Carga un modelo desde su checkpoint de screening."""
    ckpt_path = SCREENING_DIR / f"{nombre_arch}_best.pth"
    if not ckpt_path.exists():
        print(f"  [SALTAR] No se encuentra: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = create_model(arch, encoder_name=encoder, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(DEVICE).eval()
    return model


def predecir(modelo, img_bgr, umbral=0.5):
    """Predice máscara binaria."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transformado = TRANSFORM(image=img_rgb)
    tensor = transformado["image"].unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        pred = torch.sigmoid(modelo(tensor)).squeeze().cpu().numpy()

    mascara = (pred > umbral).astype(np.uint8) * 255
    mascara = cv2.resize(mascara, (img_bgr.shape[1], img_bgr.shape[0]))
    return mascara, pred


def superponer(imagen, mascara, alpha=0.4, color=COLOR_OVERLAY):
    """Overlay de la máscara sobre la imagen."""
    overlay = imagen.copy()
    overlay[mascara > 0] = color
    return cv2.addWeighted(imagen, 1 - alpha, overlay, alpha, 0)


def calcular_estadisticas(mascara):
    """Calcula estadísticas de la máscara predicha."""
    total_px = mascara.size
    pos_px = int((mascara > 0).sum())
    pct = (pos_px / total_px) * 100
    return {
        "pixeles_herida": pos_px,
        "area_pct": round(pct, 4),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comparar 12 modelos del screening")
    parser.add_argument("--imagen", type=Path, default=DEFAULT_IMAGEN,
                        help="Ruta a la imagen a analizar")
    parser.add_argument("--salida", type=Path, default=SALIDA_DIR,
                        help="Directorio de salida")
    parser.add_argument("--umbral", type=float, default=0.5,
                        help="Umbral de binarización (default: 0.5)")
    args = parser.parse_args()

    if not args.imagen.exists():
        print(f"[ERROR] No se encuentra la imagen: {args.imagen}")
        sys.exit(1)

    SALIDA = args.salida
    SALIDA.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"COMPARACIÓN DE {len(MODELOS)} MODELOS — SCREENING")
    print("=" * 80)
    print(f"Imagen:  {args.imagen}")
    print(f"Device:  {DEVICE.upper()}")
    print(f"Umbral:  {args.umbral}")
    print(f"Salida:  {SALIDA}")
    print("=" * 80)

    # Cargar imagen
    original = cv2.imread(str(args.imagen))
    if original is None:
        print(f"[ERROR] No se pudo leer la imagen: {args.imagen}")
        sys.exit(1)

    print(f"Resolución original: {original.shape[1]}x{original.shape[0]}")
    cv2.imwrite(str(SALIDA / "00_original.png"), original)
    print()

    # Resultados
    resultados = []

    for nombre_arch, arch, encoder in MODELOS:
        print(f"[{nombre_arch:40s}] ", end="", flush=True)

        t0 = time.time()

        modelo = cargar_modelo(nombre_arch, arch, encoder)
        if modelo is None:
            print("CHECKPOINT NO ENCONTRADO")
            continue

        # Contar parámetros
        params = sum(p.numel() for p in modelo.parameters())

        # Predecir
        mascara, pred_raw = predecir(modelo, original, umbral=args.umbral)
        stats = calcular_estadisticas(mascara)

        # Guardar overlay
        overlay = superponer(original, mascara)
        overlay_path = SALIDA / f"{nombre_arch}_superpuesto.png"
        cv2.imwrite(str(overlay_path), overlay)

        # Guardar máscara sola
        mask_path = SALIDA / f"{nombre_arch}_mascara.png"
        cv2.imwrite(str(mask_path), mascara)

        elapsed = time.time() - t0

        resultados.append({
            "modelo": nombre_arch,
            "arquitectura": arch,
            "encoder": encoder,
            "params": params,
            "area_herida_pct": stats["area_pct"],
            "pixeles_herida": stats["pixeles_herida"],
            "tiempo_inferencia_s": round(elapsed, 3),
        })

        print(f"herida: {stats['area_pct']:6.2f}%  |  {elapsed:.2f}s")

    # Guardar CSV de resultados
    csv_path = SALIDA / "comparacion_resultados.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "modelo", "arquitectura", "encoder", "params",
            "area_herida_pct", "pixeles_herida", "tiempo_inferencia_s"
        ])
        writer.writeheader()
        for r in resultados:
            writer.writerow(r)

    # Mostrar ranking
    print("\n" + "=" * 80)
    print("RANKING POR ÁREA DETECTADA")
    print("=" * 80)
    resultados.sort(key=lambda r: r["area_herida_pct"], reverse=True)
    for i, r in enumerate(resultados, 1):
        print(f"  {i:2d}. {r['modelo']:35s} | herida: {r['area_herida_pct']:6.2f}% | "
              f"params: {r['params']/1e6:.1f}M | {r['tiempo_inferencia_s']:.2f}s")

    print(f"\nResultados guardados en: {SALIDA}")
    print(f"  {len(resultados)} overlays + {csv_path.name}")


if __name__ == "__main__":
    main()
