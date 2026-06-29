#!/usr/bin/env python3
"""Genera dataset YOLO desde las mascaras de segmentacion existentes.

Uso:
    python scripts/pipeline/8_prepare_yolo.py                         # solo bbox (original)
    python scripts/pipeline/8_prepare_yolo.py --segment                # bbox + poligonos segmentacion
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT = Path(__file__).parent.parent.parent
YOLO_DIR = PROJECT / "data" / "yolo"


def generate_bbox(mask_path: Path):
    """Extrae bounding box desde mascara. Retorna None si no hay herida."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or (mask > 127).sum() < 100:
        return None
    ys, xs = np.where(mask > 127)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h, w = mask.shape
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return (cx, cy, bw, bh)


def generate_polygons(
    mask_path: Path,
    epsilon_ratio: float = 0.001,
    max_points: int = 32,
) -> list[np.ndarray] | None:
    """Extract normalized polygon contours from a binary mask.

    For each connected component in the mask, extracts the outer contour
    via cv2.findContours, approximates it with cv2.approxPolyDP, and
    normalizes coordinates to [0, 1].

    Args:
        mask_path: Path to binary mask image.
        epsilon_ratio: Fraction of perimeter for contour approximation
            (default 0.001).
        max_points: Maximum polygon points per contour (default 32).

    Returns:
        List of (N, 2) float64 arrays with normalized coordinates,
        or None if the mask is empty / invalid.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    wound_px = int((mask_bin > 0).sum())
    if wound_px < 100:
        return None

    h, w = mask.shape
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: list[np.ndarray] = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1.0:
            continue
        epsilon = epsilon_ratio * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        points = approx.squeeze()
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            continue  # degenerate contour

        # Subsample if too many vertices
        if points.shape[0] > max_points:
            step = max(1, points.shape[0] // max_points)
            points = points[::step]

        # Normalize to [0, 1]
        norm = points.astype(np.float64) / [w, h]
        polygons.append(norm)

    return polygons if polygons else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generar dataset YOLO desde mascaras de segmentacion",
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Ademas del bbox, generar etiquetas de segmentacion (poligonos) en data/yolo/segment/",
    )
    args = parser.parse_args()

    # ── Crear directorios ──────────────────────────────────────────
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        (YOLO_DIR / d).mkdir(parents=True, exist_ok=True)

    if args.segment:
        for d in ["images/train", "images/val", "labels/train", "labels/val"]:
            (YOLO_DIR / "segment" / d).mkdir(parents=True, exist_ok=True)

    # ── Leer y dividir dataset ──────────────────────────────────────
    df = pd.read_csv(PROJECT / "data" / "processed" / "dataset_final.csv")
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_idx = train_test_split(range(len(train_df)), test_size=0.15, random_state=42)[1]
    train_idx = [i for i in range(len(train_df)) if i not in val_idx]
    train_split = train_df.iloc[train_idx]
    val_split = train_df.iloc[val_idx]

    print(f"Train: {len(train_split)} | Val: {len(val_split)}")

    # ── Procesar cada muestra ───────────────────────────────────────
    bbox_count = 0
    poly_count = 0
    empty_warnings = []

    for split_name, subset in [("train", train_split), ("val", val_split)]:
        for _, row in subset.iterrows():
            img_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"])
            img_name = img_path.stem
            mask_name = mask_path.stem

            # Copy image (shared between bbox and segment)
            dest_img = YOLO_DIR / "images" / split_name / img_path.name
            if not dest_img.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    cv2.imwrite(str(dest_img), img)

            # Also copy to segment/images/ for --segment
            if args.segment:
                seg_img = YOLO_DIR / "segment" / "images" / split_name / img_path.name
                if not seg_img.exists():
                    img_cv = cv2.imread(str(img_path))
                    if img_cv is not None:
                        cv2.imwrite(str(seg_img), img_cv)

            # ── Bounding box label (always) ─────────────────────────
            bbox = generate_bbox(mask_path)
            dest_label = YOLO_DIR / "labels" / split_name / f"{img_name}.txt"
            if bbox:
                with open(dest_label, "w") as f:
                    f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                bbox_count += 1

            # ── Segment polygon labels (only with --segment) ────────
            if args.segment:
                polygons = generate_polygons(mask_path)
                seg_label = YOLO_DIR / "segment" / "labels" / split_name / f"{img_name}.txt"
                if polygons:
                    with open(seg_label, "w") as f:
                        for poly in polygons:
                            flat = poly.flatten()
                            line = "0 " + " ".join(f"{v:.6f}" for v in flat)
                            f.write(line + "\n")
                    poly_count += 1
                else:
                    empty_warnings.append(mask_name)
                    print(f"  WARNING: {mask_name} has empty mask — skipping")

    print(f"\nBounding boxes generados: {bbox_count}")
    if args.segment:
        print(f"Poligonos de segmentacion generados: {poly_count}")
        if empty_warnings:
            print(f"Mascaras vacias omitidas: {len(empty_warnings)}")

    # ── Crear data.yaml ─────────────────────────────────────────────
    yaml_content = f"""path: {YOLO_DIR.as_posix()}
train: images/train
val: images/val
nc: 1
names: ['wound']
"""
    with open(YOLO_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)
    print("data.yaml creado")

    if args.segment:
        seg_yaml_content = f"""path: {(YOLO_DIR / 'segment').as_posix()}
train: images/train
val: images/val
nc: 1
names: ['wound']
"""
        with open(YOLO_DIR / "segment" / "data.yaml", "w") as f:
            f.write(seg_yaml_content)
        print("segment/data.yaml creado")

    print(f"Dataset YOLO en: {YOLO_DIR}")


if __name__ == "__main__":
    main()
