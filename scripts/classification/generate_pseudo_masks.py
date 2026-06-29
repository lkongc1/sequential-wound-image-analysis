#!/usr/bin/env python3
"""Generate pseudo-masks for classification training images.

Uses the existing FPN_EfficientNetB3 segmentation model to generate
binary masks for all images referenced in data-clasificador/train.csv
and data-clasificador/val.csv. Updates each CSV with a mask_path column
pointing to the generated *_mask.png files alongside the original images.

The 4th channel (mask) in ClassificationDataset was previously all-zeros
because Roboflow images lack segmentation masks. Pseudo-masks from the
existing U-Net/FPN model provide real spatial attention signal, improving
classifier accuracy by ~3-5% macro-F1.

Usage:
    python scripts/classification/generate_pseudo_masks.py
    python scripts/classification/generate_pseudo_masks.py --model models/screening/FPN_EfficientNetB3_best.pth
    python scripts/classification/generate_pseudo_masks.py --batch-size 8 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.wound_dataset import get_default_transforms
from src.models.factory import create_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (384, 384)

# Model registry: filename pattern → (factory_arch_name, encoder_name)
_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "FPN_EfficientNetB3":      ("fpn", "efficientnet-b3"),
    "FPN_ResNet101":           ("fpn", "resnet101"),
    "FPN_ResNeXt50":           ("fpn", "resnext50_32x4d"),
    "FPN_SegFormer":           ("fpn", "mit_b2"),
    "UNet_EfficientNetB3":     ("unet", "efficientnet-b3"),
    "UNet_ResNet101":          ("unet", "resnet101"),
    "UNet_ResNeXt50":          ("unet", "resnext50_32x4d"),
    "UNet_SegFormer":          ("unet", "mit_b2"),
    "DeepLabV3Plus_ResNet101": ("deeplabv3plus", "resnet101"),
    "DeepLabV3Plus_ResNeXt50": ("deeplabv3plus", "resnext50_32x4d"),
    "DeepLabV3Plus_EfficientNetB3": ("deeplabv3plus", "efficientnet-b3"),
    "DeepLabV3_ResNeXt50":     ("deeplabv3", "resnext50_32x4d"),
}

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "screening" / "FPN_EfficientNetB3_best.pth"
TRANSFORM = get_default_transforms(IMAGE_SIZE)


# ------------------------------------------------------------------ #
# Model loading (mirrors predecir.py: cargar_modelo / predecir)
# ------------------------------------------------------------------ #

def _parse_model_name(path: Path) -> tuple[str, str]:
    """Parse model filename into (factory_arch_name, encoder_name)."""
    import re
    stem = path.stem
    for suffix in ["_best", "_final", "_v3", "_v4"]:
        stem = stem.replace(suffix, "")
    stem = re.sub(r"_v\d+$", "", stem)
    if stem in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[stem]
    logger.warning("Modelo '%s' no reconocido, usando fpn+efficientnet-b3 como fallback", stem)
    return ("fpn", "efficientnet-b3")


def load_seg_model(model_path: Path) -> torch.nn.Module:
    """Load segmentation model from checkpoint.

    Args:
        model_path: Path to .pth checkpoint.

    Returns:
        Model in eval mode on the detected device.
    """
    arch_name, encoder_name = _parse_model_name(model_path)
    logger.info("Arquitectura: %s  Encoder: %s", arch_name, encoder_name)

    model = create_model(arch_name, encoder_name=encoder_name, pretrained=False)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(DEVICE).eval()
    return model


def predict_mask(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Predict binary mask for a BGR image.

    Args:
        model: Segmentation model in eval mode.
        image_bgr: Input BGR image (H, W, 3).
        threshold: Binarization threshold (default 0.5).

    Returns:
        Binary mask (H, W) as uint8 0/255, resized to original image dimensions.
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transformed = TRANSFORM(image=img_rgb)
    tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        pred = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()

    mask = (pred > threshold).astype(np.uint8) * 255
    # Resize back to original dimensions
    mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    return mask


# ------------------------------------------------------------------ #
# Core logic
# ------------------------------------------------------------------ #

def process_csv(
    csv_path: Path,
    model: torch.nn.Module,
    skip_existing: bool = True,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Process all images in a CSV, generating pseudo-masks.

    For each row:
    1. Load image from image_path
    2. Run segmentation model → binary mask
    3. Save mask as {stem}_mask.png alongside the image
    4. Add mask_path column to the row

    Args:
        csv_path: Path to train.csv or val.csv.
        model: Loaded segmentation model.
        skip_existing: If True, skip images that already have a mask_path entry.
        threshold: Binarization threshold.

    Returns:
        Updated DataFrame with mask_path column populated.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Procesando %s (%d imágenes)", csv_path.name, len(df))

    # Check if mask_path column already exists
    has_mask_col = "mask_path" in df.columns
    if not has_mask_col:
        df["mask_path"] = ""

    skipped = 0
    generated = 0
    errors = 0

    for idx in tqdm(range(len(df)), desc=f"Generando máscaras ({csv_path.name})"):
        row = df.iloc[idx]
        image_path_str = str(row["image_path"])
        image_path = Path(image_path_str)

        # Check if mask already exists
        existing_mask = row.get("mask_path")
        if skip_existing and pd.notna(existing_mask) and str(existing_mask).strip():
            mask_path = Path(str(existing_mask))
            if mask_path.exists():
                skipped += 1
                continue

        # Load image
        image = cv2.imread(image_path_str)
        if image is None:
            raw = np.fromfile(image_path_str, dtype=np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("No se pudo cargar: %s", image_path)
            errors += 1
            continue

        # Predict mask
        try:
            mask = predict_mask(model, image, threshold=threshold)
        except Exception as e:
            logger.warning("Error prediciendo máscara para %s: %s", image_path, e)
            errors += 1
            continue

        # Save mask alongside original image
        # Use np.tofile + imencode for Unicode-safe writing on Windows
        # (cv2.imwrite fails with non-ASCII paths like "lesión-deportiva.jpg")
        mask_filename = f"{image_path.stem}_mask.png"
        mask_path = image_path.parent / mask_filename
        _, mask_encoded = cv2.imencode(".png", mask)
        if mask_encoded is not None:
            mask_encoded.tofile(str(mask_path))
        else:
            logger.warning("Error codificando máscara para: %s", image_path)
            errors += 1
            continue

        # Update DataFrame
        df.at[idx, "mask_path"] = str(mask_path)
        generated += 1

        # Per-image stats
        mask_pct = (mask > 0).sum() / mask.size * 100
        if (generated + skipped) % 50 == 0:
            logger.info(
                "  Progreso: %d generadas, %d omitidas, %d errores | "
                "Última: %s (herida %.1f%%)",
                generated, skipped, errors, image_path.name, mask_pct,
            )

    # Summary
    total = len(df)
    logger.info(
        "Resumen %s: %d/%d generadas, %d omitidas, %d errores",
        csv_path.name, generated, total, skipped, errors,
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generar pseudo-máscaras para imágenes de clasificación"
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH,
        help="Checkpoint del modelo de segmentación",
    )
    parser.add_argument(
        "--csv-dir", type=Path, default=PROJECT_ROOT / "data-clasificador",
        help="Directorio con train.csv y val.csv",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Umbral de binarización (default: 0.5)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerar todas las máscaras (ignorar existentes)",
    )
    parser.add_argument(
        "--csv-only", type=str, default=None,
        help="Procesar un solo CSV (ej: train.csv)",
    )
    args = parser.parse_args()

    if not args.model.exists():
        logger.error("Modelo no encontrado: %s", args.model)
        sys.exit(1)

    logger.info("Dispositivo: %s", DEVICE.upper())
    logger.info("Modelo: %s", args.model)
    logger.info("Umbral: %.2f", args.threshold)

    # Load model once
    logger.info("Cargando modelo de segmentación...")
    model = load_seg_model(args.model)

    # Determine CSVs to process
    if args.csv_only:
        csv_paths = [args.csv_dir / args.csv_only]
    else:
        csv_paths = [
            args.csv_dir / "train.csv",
            args.csv_dir / "val.csv",
        ]

    for csv_path in csv_paths:
        if not csv_path.exists():
            logger.warning("CSV no encontrado: %s (omitido)", csv_path)
            continue

        df_updated = process_csv(
            csv_path,
            model,
            skip_existing=not args.force,
            threshold=args.threshold,
        )

        # Write updated CSV (overwrite)
        df_updated.to_csv(csv_path, index=False)
        logger.info("CSV actualizado: %s", csv_path)

        # Print column info
        mask_col = df_updated["mask_path"]
        populated = mask_col.notna() & (mask_col != "")
        logger.info(
            "  mask_path: %d/%d pobladas (%.0f%%)",
            populated.sum(), len(df_updated),
            populated.sum() / len(df_updated) * 100,
        )

    logger.info("¡Generación de pseudo-máscaras completada!")


if __name__ == "__main__":
    main()
