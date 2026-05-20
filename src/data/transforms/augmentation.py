"""Data augmentation with albumentations (Pylance-clean version)."""
from __future__ import annotations

import albumentations as A
import cv2


def get_training_augmentation(image_size: int = 384) -> A.Compose:
    """Standard training augmentation pipeline.

    Uses HorizontalFlip/VerticalFlip (not deprecated Flip).
    Uses var_limit for GaussNoise (not noise_limit).
    Uses tuples for Normalize mean/std (not lists).
    """
    return A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Affine(
                translate_percent=(-0.1, 0.1),
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.GaussNoise(
                std_range=(0.01, 0.05),
                mean_range=(0.0, 0.0),
                p=0.3,
            ),
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def get_inference_augmentation() -> A.Compose:
    """Minimal augmentation for inference (only normalization)."""
    return A.Compose(
        [
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )