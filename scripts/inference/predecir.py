#!/usr/bin/env python3
"""Prueba local del modelo de segmentacion de heridas.

Predice la mascara para una imagen y guarda:
  - original.png
  - mascara.png
  - superpuesto.png (overlay)

Uso:
    python scripts/inference/predecir.py imagen.png
    python scripts/inference/predecir.py imagen.png --umbral 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch
from PIL import Image

from src.datasets.wound_dataset import get_default_transforms
from src.models.factory import create_model

MODEL_PATH = PROJECT_ROOT / "models" / "unet_final_pretrained.pth"
IMAGE_SIZE = (256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRANSFORM = get_default_transforms(IMAGE_SIZE)


def cargar_modelo() -> torch.nn.Module:
    modelo = create_model("unet", pretrained=False)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    modelo.load_state_dict(ckpt["model_state_dict"], strict=False)
    modelo.to(DEVICE).eval()
    return modelo


def predecir(modelo: torch.nn.Module, imagen: np.ndarray, umbral: float = 0.5) -> np.ndarray:
    """Predice mascara binaria para una imagen BGR (H, W, 3)."""
    img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    transformado = TRANSFORM(image=img_rgb)
    tensor = transformado["image"].unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        pred = torch.sigmoid(modelo(tensor)).squeeze().cpu().numpy()

    mascara = (pred > umbral).astype(np.uint8) * 255
    mascara = cv2.resize(mascara, (imagen.shape[1], imagen.shape[0]))
    return mascara


def superponer(imagen: np.ndarray, mascara: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay rojo sobre la region detectada."""
    overlay = imagen.copy()
    overlay[mascara > 0] = [0, 0, 255]
    return cv2.addWeighted(imagen, 1 - alpha, overlay, alpha, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probar segmentacion de heridas en una imagen local")
    parser.add_argument("imagen", type=Path, help="Ruta a la imagen a analizar")
    parser.add_argument("--umbral", type=float, default=0.5, help="Umbral de binarizacion (default: 0.5)")
    parser.add_argument("--salida", type=Path, default=None, help="Directorio de salida (default: junto a la imagen)")
    args = parser.parse_args()

    if not args.imagen.exists():
        print(f"[ERROR] No se encontro: {args.imagen}", file=sys.stderr)
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"[ERROR] Modelo no encontrado: {MODEL_PATH}", file=sys.stderr)
        print(f"        Ejecuta primero: python scripts/6_train_unet_final.py", file=sys.stderr)
        sys.exit(1)

    salida = args.salida or args.imagen.parent
    salida.mkdir(parents=True, exist_ok=True)

    nombre = args.imagen.stem

    print(f"Imagen:  {args.imagen}")
    print(f"Modelo:  {MODEL_PATH.name}")
    print(f"Device:  {DEVICE.upper()}")
    print(f"Umbral:  {args.umbral}")
    print()

    print("Cargando modelo...")
    modelo = cargar_modelo()

    print("Leyendo imagen...")
    original = cv2.imread(str(args.imagen))
    if original is None:
        print(f"[ERROR] No se pudo leer la imagen: {args.imagen}", file=sys.stderr)
        sys.exit(1)

    print(f"Resolucion original: {original.shape[1]}x{original.shape[0]}")

    print("Prediciendo mascara...")
    mascara = predecir(modelo, original, args.umbral)

    herida_pct = (mascara > 0).sum() / mascara.size * 100
    print(f"Herida detectada: {herida_pct:.2f}% de la imagen")

    print("Guardando resultados...")
    cv2.imwrite(str(salida / f"{nombre}_original.png"), original)
    cv2.imwrite(str(salida / f"{nombre}_mascara.png"), mascara)
    cv2.imwrite(str(salida / f"{nombre}_superpuesto.png"), superponer(original, mascara))

    print(f"\nResultados guardados en: {salida}/")
    print(f"  {nombre}_original.png")
    print(f"  {nombre}_mascara.png")
    print(f"  {nombre}_superpuesto.png")


if __name__ == "__main__":
    main()
