#!/usr/bin/env python3
"""Download, normalize, balance, and split 7-class wound classification data.

Downloads 3 Roboflow datasets into data-clasificador/raw/, maps English labels
to Spanish taxonomy, undersamples to ~600/class, and produces a stratified
70/15/15 train/val/test split.

Usage:
    python scripts/download_classification_data.py
    python scripts/download_classification_data.py --skip-download  # use cached data only
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

API_KEY: str = "DVnQXr1udAOhcikIJWnJ"
OUTPUT_DIR: Path = Path("data-clasificador")
RAW_DIR: Path = OUTPUT_DIR / "raw"
MANIFEST_PATH: Path = OUTPUT_DIR / "manifest.csv"
SEED: int = 42
TARGET_PER_CLASS: int = 600

# Label mapping: English (Roboflow label) → Spanish (7-class taxonomy)
LABEL_MAP: Dict[str, str] = {
    "abrasion": "raspón",
    "bruise": "hematoma",
    "burn": "quemadura",
    "cut": "corte",
    "laceration": "laceración",
    "stab": "punción",
    "normal_skin": "piel_sana",
}

# Ordered Spanish class names
CLASS_NAMES: List[str] = [
    "raspón",
    "hematoma",
    "quemadura",
    "corte",
    "laceración",
    "punción",
    "piel_sana",
]

# Roboflow datasets to download.
# Each entry: (workspace, project_name, version_number, label)
# Adjust workspace/project/version to match your Roboflow account.
DATASETS: List[Tuple[str, str, int, str]] = [
    ("wound-segmentation", "smartheal", 1, "SmartHeal"),
    ("wound-segmentation", "basic-wound-classify", 1, "BasicWoundClassify"),
    ("wound-segmentation", "burn-wound-classification", 1, "BurnWound"),
]

# ------------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Download
# ------------------------------------------------------------------ #


def download_datasets(skip_download: bool = False) -> None:
    """Download all configured Roboflow datasets into RAW_DIR.

    Skips download if the dataset directory already exists and contains images,
    unless *skip_download* is False (default). When network is unavailable but
    local cache exists, a warning is printed and the cache is used.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error(
            "Roboflow SDK not installed. Run: pip install roboflow"
        )
        sys.exit(1)

    rf = Roboflow(api_key=API_KEY)

    for workspace_name, project_name, version_num, label in DATASETS:
        dest_dir = RAW_DIR / label
        if dest_dir.exists() and any(dest_dir.iterdir()):
            logger.info(
                "Using cached data for %s in %s", label, dest_dir
            )
            continue

        if skip_download:
            if dest_dir.exists():
                logger.info("Skip-download: using cached %s", label)
                continue
            logger.warning(
                "Skip-download: no cache for %s — skipping", label
            )
            continue

        logger.info(
            "Downloading %s/%s v%d → %s",
            workspace_name, project_name, version_num, dest_dir,
        )
        try:
            workspace = rf.workspace(workspace_name)
            project = workspace.project(project_name)
            version = project.version(version_num)
            dataset = version.download("folder", location=str(dest_dir))
            logger.info("Downloaded %d images to %s", len(dataset), dest_dir)
        except Exception as exc:
            logger.error(
                "Failed to download %s/%s: %s", workspace_name, project_name, exc
            )
            logger.error(
                "Check that workspace '%s', project '%s', version %d exist "
                "and your API key has access.",
                workspace_name, project_name, version_num,
            )
            if not any(dest_dir.iterdir()) if dest_dir.exists() else True:
                raise


# ------------------------------------------------------------------ #
# Scan downloaded data
# ------------------------------------------------------------------ #


def scan_raw_data() -> pd.DataFrame:
    """Scan RAW_DIR for images and extract labels from subdirectory names.

    Assumes each dataset is stored as:
        data-clasificador/raw/{DatasetName}/{className}/*.jpg

    Returns:
        DataFrame with columns: image_path, label, source
    """
    records: List[Dict[str, str]] = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    for dataset_dir in sorted(RAW_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        source = dataset_dir.name
        for class_dir in sorted(dataset_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            original_label = class_dir.name.lower().replace(" ", "_")
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in image_extensions:
                    records.append(
                        {
                            "image_path": str(img_file.resolve()),
                            "label": original_label,
                            "source": source,
                        }
                    )

    if not records:
        logger.error(
            "No images found in %s. Did the download succeed?", RAW_DIR
        )
        sys.exit(1)

    df = pd.DataFrame(records)
    logger.info("Scanned %d images from %d sources", len(df), df["source"].nunique())
    logger.info("Raw label distribution:\n%s", df["label"].value_counts().to_string())
    return df


# ------------------------------------------------------------------ #
# Normalize labels
# ------------------------------------------------------------------ #


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map English Roboflow labels to Spanish 7-class taxonomy.

    Raises:
        SystemExit: if any label is not in LABEL_MAP.
    """
    original_labels = set(df["label"].unique())
    unknown = original_labels - set(LABEL_MAP.keys())

    if unknown:
        logger.error("Unknown labels found: %s", sorted(unknown))
        logger.error(
            "These labels are not in the mapping table. "
            "Add them to LABEL_MAP (English→Spanish) or exclude "
            "those classes from the dataset."
        )
        sys.exit(1)

    df["label"] = df["label"].map(LABEL_MAP)
    logger.info(
        "Normalized %d images. Class distribution:\n%s",
        len(df),
        df["label"].value_counts().to_string(),
    )
    return df


# ------------------------------------------------------------------ #
# Balance
# ------------------------------------------------------------------ #


def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Randomly undersample majority classes to TARGET_PER_CLASS.

    Classes below the target are kept at their original size. Sampling uses
    a fixed seed (SEED) for reproducibility.
    """
    rng = random.Random(SEED)
    balanced_parts: List[pd.DataFrame] = []

    logger.info("Balancing classes (target ~%d/class, seed=%d):", TARGET_PER_CLASS, SEED)
    for class_name in CLASS_NAMES:
        class_df = df[df["label"] == class_name]
        count = len(class_df)

        if count == 0:
            logger.warning("  %s: 0 images — class missing from datasets", class_name)
            continue

        if count > TARGET_PER_CLASS:
            sampled = class_df.sample(n=TARGET_PER_CLASS, random_state=SEED)
            balanced_parts.append(sampled)
            logger.info(
                "  %s: %d → %d (undersampled)", class_name, count, TARGET_PER_CLASS
            )
        else:
            balanced_parts.append(class_df)
            logger.info("  %s: %d (below target, kept all)", class_name, count)

    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    logger.info("Balanced total: %d images", len(balanced_df))
    return balanced_df


# ------------------------------------------------------------------ #
# Stratified split
# ------------------------------------------------------------------ #


def stratified_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified 70/15/15 train/val/test split.

    Uses sklearn.train_test_split with random_state=SEED. Stratification
    preserves the per-class distribution across splits.
    """
    # First split: 70 train, 30 temp (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=SEED,
    )

    # Second split: 15 val, 15 test (50/50 of temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=SEED,
    )

    logger.info("Split sizes: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logger.info(
            "%s per-class:\n%s",
            name,
            split_df["label"].value_counts().to_string(),
        )

    return train_df, val_df, test_df


# ------------------------------------------------------------------ #
# Write outputs
# ------------------------------------------------------------------ #


def write_split_csvs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Write train.csv, val.csv, test.csv with image_path and label columns."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, split_df in splits.items():
        out_path = OUTPUT_DIR / f"{name}.csv"
        split_df[["image_path", "label"]].to_csv(out_path, index=False)
        logger.info("Wrote %s (%d samples)", out_path, len(split_df))


def write_manifest(df: pd.DataFrame) -> None:
    """Write manifest CSV with source dataset and per-class counts."""
    manifest = (
        df.groupby(["source", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["source", "label"])
    )
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    logger.info("Wrote manifest to %s", MANIFEST_PATH)


# ------------------------------------------------------------------ #
# Verify
# ------------------------------------------------------------------ #


def verify_outputs() -> bool:
    """Verify that output CSVs exist and have consistent labels."""
    errors = 0

    for name in ("train", "val", "test"):
        path = OUTPUT_DIR / f"{name}.csv"
        if not path.exists():
            logger.error("Missing output: %s", path)
            errors += 1
            continue

        df = pd.read_csv(path)
        labels = set(df["label"].unique())
        unknown = labels - set(CLASS_NAMES)
        if unknown:
            logger.error("%s contains unknown labels: %s", path, unknown)
            errors += 1

        missing = set(CLASS_NAMES) - labels
        if missing:
            logger.warning("%s missing classes: %s", path, missing)

        logger.info(
            "%s: %d samples, classes=%s",
            name,
            len(df),
            sorted(labels),
        )

    if not MANIFEST_PATH.exists():
        logger.error("Missing manifest: %s", MANIFEST_PATH)
        errors += 1

    return errors == 0


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare 7-class wound classification data"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use cached data only, skip Roboflow download",
    )
    parser.add_argument(
        "--skip-balance",
        action="store_true",
        help="Skip class balancing step",
    )
    args = parser.parse_args()

    # 1. Download
    download_datasets(skip_download=args.skip_download)

    # 2. Scan
    df = scan_raw_data()

    # 3. Normalize labels
    df = normalize_labels(df)

    # 4. Write manifest
    write_manifest(df)

    # 5. Balance
    if not args.skip_balance:
        df = balance_classes(df)

    # 6. Split
    train_df, val_df, test_df = stratified_split(df)

    # 7. Write CSVs
    write_split_csvs(train_df, val_df, test_df)

    # 8. Verify
    ok = verify_outputs()
    if ok:
        logger.info("Pipeline completed successfully.")
    else:
        logger.error("Pipeline completed with errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
