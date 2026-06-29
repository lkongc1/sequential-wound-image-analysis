"""Wound dataset with image/mask loading and transforms (Pylance-clean)."""
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


def get_default_transforms(image_size: Tuple[int, int] = (256, 256)) -> A.Compose:
    """Get default transforms: resize and normalization (no augmentation).

    Args:
        image_size: target (height, width) for resizing

    Returns:
        Albumentations Compose with resize and normalization
    """
    return A.Compose(
        [
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


def get_training_transforms(image_size: Tuple[int, int] = (384, 384)) -> A.Compose:
    """Get training transforms with heavy data augmentation.

    Augmentations are applied in sync to both image and mask.
    Designed to desensitize the model against:
      - Blood stains, scabs, suture marks (ElasticTransform, GridDistortion, GaussNoise)
      - Reddish skin tones being confused with wounds (HueSaturationValue)
      - Small dataset overfitting (spatial transforms)

    Order: spatial -> color -> noise -> resize -> normalize -> tensor

    Args:
        image_size: target (height, width) for resizing

    Returns:
        Albumentations Compose with full augmentation pipeline
    """
    return A.Compose(
        [
            # --- Spatial (applied to image AND mask) ---
            A.ElasticTransform(
                alpha=30, sigma=5, p=0.4,
            ),
            A.GridDistortion(
                num_steps=5, distort_limit=0.15, p=0.3,
            ),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),

            # --- Color / texture (image only, mask untouched) ---
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.4,
            ),

            # --- Noise (desensitize against fine texture) ---
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),

            # --- Finalize ---
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


def load_image_safely(path: Path) -> np.ndarray:
    """Load image with validation and error logging for FDA traceability.

    On Windows, cv2.imread(str(path)) fails with non-ASCII paths like
    "lesión-deportiva.jpg".  We use np.fromfile + cv2.imdecode to handle
    unicode paths reliably across platforms.
    """
    try:
        image = cv2.imread(str(path))
        if image is None:
            # Fallback: read raw bytes and decode (handles unicode paths on Windows)
            raw = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except Exception:
        raw = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    if image is None:
        logger.error(f"Failed to load image: {path}")
        raise FileNotFoundError(f"Image not found or corrupted: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask_safely(path: Path) -> np.ndarray:
    """Load binary mask with validation and error logging.

    Uses np.fromfile + cv2.imdecode for unicode path safety on Windows.
    """
    try:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raw = np.fromfile(str(path), dtype=np.uint8)
            mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    except Exception:
        raw = np.fromfile(str(path), dtype=np.uint8)
        mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        logger.error(f"Failed to load mask: {path}")
        raise FileNotFoundError(f"Mask not found or corrupted: {path}")
    return (mask > 127).astype(np.float32)


class WoundDataset(Dataset):
    """Dataset for wound segmentation with data augmentation."""

    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        transform: Optional[A.Compose] = None,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple:
        image = load_image_safely(self.image_paths[idx])
        mask = load_mask_safely(self.mask_paths[idx])

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            # Convert numpy arrays to tensors
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=0)
            mask = torch.from_numpy(mask)

        # Ensure mask always has channel dimension (1, H, W)
        if isinstance(mask, torch.Tensor) and mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return image, mask


class WoundSegmentationDataset(Dataset):
    """Simple dataset from image/mask directories for evaluation."""

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_size: Tuple[int, int] = (384, 384),
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.transform = A.Compose(
            [
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ],
            additional_targets={"mask": "mask"},
        )
        self.image_files = sorted(
            [f for f in self.images_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_files[idx]
        mask_path = self.masks_dir / image_path.name
        image = load_image_safely(image_path)
        mask = load_mask_safely(mask_path)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).unsqueeze(0)

        return {"image": image, "mask": mask, "filename": image_path.name}


def create_dataset_from_csv(
    csv_path: Path | str,
    split: str = "train",
    image_size: Tuple[int, int] = (256, 256),
    transform: Optional[A.Compose] = None,
) -> WoundDataset:
    """Factory function to create WoundDataset from cleaned CSV file (SRP).

    Args:
        csv_path: path to dataset_cleaned.csv
        split: filter by split column ("train", "val", "test")
        image_size: target (height, width) for resizing
        transform: optional albumentations transform. If None, uses default.

    Returns:
        WoundDataset instance

    Raises:
        FileNotFoundError: if CSV doesn't exist
        ValueError: if no samples found for given split
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter by split if column exists
    if "split" in df.columns:
        df = df[df["split"] == split]

    # Filter out outliers and empty masks
    if "is_outlier" in df.columns:
        df = df[~df["is_outlier"]]
    if "is_empty" in df.columns:
        df = df[~df["is_empty"]]

    if len(df) == 0:
        raise ValueError(f"No samples found for split='{split}' in {csv_path}")

    image_paths = [Path(p) for p in df["image_path"]]
    mask_paths = [Path(p) for p in df["mask_path"]]

    if transform is None:
        transform = get_default_transforms(image_size)

    logger.info(f"Created WoundDataset with {len(df)} samples for split='{split}'")
    return WoundDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        transform=transform,
    )
