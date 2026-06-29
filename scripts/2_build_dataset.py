#!/usr/bin/env python3
"""Construye el dataset desde las imagenes raw.

Escanea train_images/ + train_masks/ y test_images/ + test_masks/.
Empareja cada imagen con su mascara, calcula estadisticas (brillo,
contraste, porcentaje de herida, area), filtra muestras invalidas
y aplica split patient-aware. Genera data/processed/dataset_final.csv

Uso:
    python scripts/2_build_dataset.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# CONFIG
# ------------------------------------------------------------------ #
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "data_wound_seg"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "dataset_final.csv"
MIN_AREA = 200
MIN_WOUND_PCT = 0.05
TEST_SPLIT = 0.20


# ------------------------------------------------------------------ #
# HELPERS
# ------------------------------------------------------------------ #

def extract_source(filename: str) -> str:
    for prefix in ["fusc", "medetec", "wsnet"]:
        if filename.lower().startswith(prefix):
            return prefix
    return "unknown"


def extract_group(filename: str) -> str:
    """Patient-aware: extrae grupo del nombre de archivo."""
    name = Path(filename).stem
    for prefix in ["fusc", "medetec", "wsnet"]:
        if name.lower().startswith(prefix):
            parts = name.split("_")
            if len(parts) >= 2:
                return f"{prefix}_{parts[1]}"
    return name


def compute_stats(img_path: Path, mask_path: Path) -> dict:
    """Calcula todas las estadisticas para un par imagen-mascara."""
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_bin = (mask > 127).astype(np.uint8)
    total_px = mask.shape[0] * mask.shape[1]

    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())
    contrast_rms = float(np.sqrt(np.mean((gray.astype(np.float64) - brightness_mean) ** 2)))

    wound_px = int(mask_bin.sum())
    wound_pct = wound_px / total_px * 100
    is_empty = wound_px == 0
    is_outlier = wound_pct > 50 or wound_pct < MIN_WOUND_PCT
    outlier_reason = ""

    if wound_pct > 50:
        outlier_reason = "wound > 50% (possible non-wound image or extreme close-up)"
    elif wound_pct < MIN_WOUND_PCT:
        outlier_reason = f"wound < {MIN_WOUND_PCT}% (too small)"

    # Edge density
    edges = cv2.Canny(mask_bin, 50, 150)
    edge_density = float(edges.sum()) / total_px if total_px > 0 else 0.0

    return {
        "filename": img_path.name,
        "source": extract_source(img_path.name),
        "image_path": str(img_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "wound_percentage": round(wound_pct, 4),
        "brightness_mean": round(brightness_mean, 2),
        "brightness_std": round(brightness_std, 2),
        "contrast_rms": round(contrast_rms, 2),
        "mask_area_pixels": wound_px,
        "mask_edge_density": round(edge_density, 6),
        "is_empty": is_empty,
        "is_outlier": is_outlier,
        "outlier_reason": outlier_reason,
        "review_status": "",
    }


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #

def main() -> None:
    logger.info("=" * 55)
    logger.info("  CONSTRUCCION DE DATASET — desde raw data")
    logger.info("=" * 55)

    rows = []

    # Procesar train y test
    for split_name in ["train", "test"]:
        img_dir = RAW_DIR / f"{split_name}_images"
        mask_dir = RAW_DIR / f"{split_name}_masks"

        if not img_dir.exists():
            logger.error("Directorio no encontrado: %s", img_dir)
            continue

        images = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
        logger.info("  %s: %d imagenes", split_name, len(images))

        for img_path in tqdm(images, desc=f"  Procesando {split_name}"):
            mask_name = img_path.stem + ".png"
            mask_path = mask_dir / mask_name

            if not mask_path.exists():
                logger.warning("Mascara faltante: %s", mask_name)
                continue

            stats = compute_stats(img_path, mask_path)
            if stats is None:
                logger.warning("Error procesando: %s", img_path.name)
                continue

            stats["split"] = split_name
            rows.append(stats)

    df = pd.DataFrame(rows)
    logger.info("  Total muestras raw: %d", len(df))

    # --- FILTRADO ---
    before = len(df)
    empty_mask = df["is_empty"] == True
    small_area = df["mask_area_pixels"] < MIN_AREA
    rejected = empty_mask | small_area

    df_clean = df[~rejected].copy()
    logger.info("  Tras filtrado: %d (eliminadas %d)", len(df_clean), before - len(df_clean))
    logger.info("    - Mascaras vacias: %d", empty_mask.sum())
    logger.info("    - Area < %d px: %d", MIN_AREA, small_area.sum())

    # --- SPLIT PATIENT-AWARE ---
    df_clean["group"] = df_clean["filename"].apply(extract_group)
    groups = df_clean["group"].unique()
    np.random.seed(42)
    np.random.shuffle(groups)

    n_test = max(1, int(len(groups) * TEST_SPLIT))
    test_groups = set(groups[:n_test])
    train_groups = set(groups[n_test:])

    df_clean["split"] = df_clean["group"].apply(
        lambda g: "test" if g in test_groups else "train"
    )

    # Guardar
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "filename", "source", "split", "image_path", "mask_path",
        "wound_percentage", "brightness_mean", "brightness_std", "contrast_rms",
        "mask_area_pixels", "mask_edge_density",
        "is_empty", "is_outlier", "outlier_reason", "review_status",
    ]
    df_clean[cols].to_csv(OUTPUT, index=False)

    logger.info("=" * 55)
    logger.info("  DATASET FINAL: %d muestras", len(df_clean))
    logger.info("    Train: %d  |  Test: %d",
                len(df_clean[df_clean["split"] == "train"]),
                len(df_clean[df_clean["split"] == "test"]))
    logger.info("    Fuentes: %s", ", ".join(sorted(df_clean["source"].unique())))
    logger.info("    Guardado: %s", OUTPUT)
    logger.info("  Siguiente: python scripts/3_eda.py")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
