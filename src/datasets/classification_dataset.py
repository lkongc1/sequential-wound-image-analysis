"""Classification dataset for 7-class wound type recognition.

Loads images from a split CSV (image_path, label columns) with optional
4-channel input (RGB + binary mask) and RandAugment transforms.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.datasets.wound_dataset import load_image_safely

logger = logging.getLogger(__name__)

# ImageNet normalization (same as segmentation pipeline)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_classification_transforms(
    image_size: Tuple[int, int] = (384, 384),
    augment: bool = False,
) -> A.Compose:
    """Build albumentations transform pipeline for classification.

    Args:
        image_size: Target (height, width).
        augment: If True, enable RandAugment-style spatial/color transforms.

    Returns:
        Albumentations Compose pipeline.
    """
    transforms: list = []

    if augment:
        # RandAugment-inspired: random spatial and color transforms
        transforms.extend([
            A.RandomResizedCrop(
                size=(image_size[0], image_size[1]),
                scale=(0.7, 1.0),
                ratio=(0.75, 1.33),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
        ])
    else:
        transforms.append(A.Resize(image_size[0], image_size[1]))

    transforms.extend([
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])

    return A.Compose(transforms)


class ClassificationDataset(Dataset):
    """PyTorch Dataset for wound type classification.

    Loads images from a split CSV with columns:
        image_path: absolute path to the image file
        label: Spanish class name (e.g. "laceración")
        mask_path: (optional) path to binary mask PNG for 4-channel mode

    Args:
        csv_path: Path to the split CSV (train.csv, val.csv, or test.csv).
        class_names: Ordered list of class label strings.
        image_size: Target (height, width) for resizing.
        use_mask: If True, returns 4-channel (RGB + mask). If False, 3-channel.
        augment: If True, enable RandAugment transforms.

    Returns:
        Tuple of (tensor, label_index) where tensor shape is (C, H, W)
        with C=3 (use_mask=False) or C=4 (use_mask=True).
    """

    def __init__(
        self,
        csv_path: Path,
        class_names: List[str],
        image_size: Tuple[int, int] = (384, 384),
        use_mask: bool = False,
        augment: bool = False,
    ) -> None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.class_names = list(class_names)
        self.image_size = image_size
        self.use_mask = use_mask
        self.augment = augment

        self._label_to_idx = {name: i for i, name in enumerate(class_names)}

        # Validate labels
        unknown = set(self.df["label"].unique()) - set(class_names)
        if unknown:
            raise ValueError(
                f"CSV contains labels not in class_names: {unknown}. "
                f"Expected: {class_names}"
            )

        # Build transform (applied per-sample since mask is conditional)
        self._transform = _get_classification_transforms(image_size, augment=augment)
        self._mask_transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
        ])

        logger.info(
            "ClassificationDataset: %d samples, use_mask=%s, augment=%s, "
            "classes=%s",
            len(self.df),
            use_mask,
            augment,
            class_names,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_path = Path(row["image_path"])
        label_str = str(row["label"])
        label_idx = self._label_to_idx[label_str]

        # Load image
        image = load_image_safely(image_path)  # (H, W, 3) RGB uint8

        # Load mask if applicable
        if self.use_mask and "mask_path" in self.df.columns:
            mask_path_raw = row.get("mask_path")
            if pd.notna(mask_path_raw) and str(mask_path_raw).strip():
                mask_path = Path(str(mask_path_raw))
                if not mask_path.exists():
                    raise FileNotFoundError(f"Mask not found: {mask_path}")
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(f"Mask corrupted: {mask_path}")
                mask = (mask > 127).astype(np.float32)
                # Resize mask
                mask_resized = self._mask_transform(image=mask)["image"]
                # mask_resized is (H, W) — add channel dim
                mask_channel = np.expand_dims(mask_resized, axis=-1)  # (H, W, 1)
            else:
                # No mask_path — use all-zeros mask channel
                mask_channel = np.zeros(
                    (self.image_size[0], self.image_size[1], 1),
                    dtype=np.float32,
                )
        elif self.use_mask:
            # use_mask=True but no mask_path column — all-zeros
            mask_channel = np.zeros(
                (self.image_size[0], self.image_size[1], 1),
                dtype=np.float32,
            )

        # Apply image transforms (image only)
        transformed = self._transform(image=image)
        image_tensor = transformed["image"]  # (3, H, W)

        if self.use_mask:
            # Convert mask channel to tensor and normalize same as image
            mask_tensor = torch.from_numpy(mask_channel).permute(2, 0, 1).float()
            # Stack into 4-channel
            image_tensor = torch.cat([image_tensor, mask_tensor], dim=0)  # (4, H, W)

        return image_tensor, label_idx
