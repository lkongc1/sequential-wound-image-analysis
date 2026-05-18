#!/usr/bin/env python3
"""Inferencia de heridas con U-Net ResNet50 pretrained.

Uso:
    python inferir.py imagen.png
    python inferir.py imagen.png --umbral 0.3

Salida:
    imagen_original.png
    imagen_mascara.png
    imagen_superpuesto.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

MODEL_PATH = Path(__file__).parent / "models" / "unet_final_pretrained.pth"
IMAGE_SIZE = (256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM = A.Compose([
    A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def cargar_modelo() -> torch.nn.Module:
    modelo = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    modelo.load_state_dict(ckpt["model_state_dict"], strict=False)
    modelo.to(DEVICE).eval()
    return modelo


def predecir(modelo: torch.nn.Module, imagen_bgr: np.ndarray, umbral: float = 0.5) -> np.ndarray:
    img_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    transformado = TRANSFORM(image=img_rgb)
    tensor = transformado["image"].unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        pred = torch.sigmoid(modelo(tensor)).squeeze().cpu().numpy()
    mascara = (pred > umbral).astype(np.uint8) * 255
    mascara = cv2.resize(mascara, (imagen_bgr.shape[1], imagen_bgr.shape[0]))
    return mascara


def superponer(imagen: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    overlay = imagen.copy()
    overlay[mascara > 0] = [0, 0, 255]
    return cv2.addWeighted(imagen, 0.6, overlay, 0.4, 0)


def main():
    parser = argparse.ArgumentParser(description="Deteccion de heridas en imagenes")
    parser.add_argument("imagen", type=Path, help="Ruta a la imagen")
    parser.add_argument("--umbral", type=float, default=0.5, help="Umbral de binarizacion (0-1)")
    args = parser.parse_args()

    if not args.imagen.exists():
        print(f"ERROR: No se encontro {args.imagen}", file=sys.stderr)
        sys.exit(1)
    if not MODEL_PATH.exists():
        print(f"ERROR: Modelo no encontrado en {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    nombre = args.imagen.stem
    salida = args.imagen.parent

    print(f"Imagen: {args.imagen}")
    print(f"Device: {DEVICE.upper()}")

    modelo = cargar_modelo()
    original = cv2.imread(str(args.imagen))
    if original is None:
        print(f"ERROR: No se pudo leer {args.imagen}", file=sys.stderr)
        sys.exit(1)

    mascara = predecir(modelo, original, args.umbral)
    pct = (mascara > 0).sum() / mascara.size * 100
    print(f"Herida detectada: {pct:.2f}%")

    cv2.imwrite(str(salida / f"{nombre}_original.png"), original)
    cv2.imwrite(str(salida / f"{nombre}_mascara.png"), mascara)
    cv2.imwrite(str(salida / f"{nombre}_superpuesto.png"), superponer(original, mascara))
    print(f"Guardado: {nombre}_mascara.png, {nombre}_superpuesto.png")


if __name__ == "__main__":
    main()
