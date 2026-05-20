"""Shared pytest fixtures for integration-style tests.

Fixtures:
    dummy_model: lightweight PyTorch segmentation model (no pretrained weights).
    synthetic_dataset: temporary directory with random image + mask PNGs.
    tmp_checkpoint: valid PyTorch checkpoint dict saved to a .pth file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def dummy_model() -> nn.Module:
    """Return a tiny CNN usable for quick forward tests."""
    return nn.Sequential(
        nn.Conv2d(3, 1, kernel_size=3, padding=1),
        nn.Sigmoid(),
    )


@pytest.fixture
def synthetic_dataset(tmp_path: Path):
    """Create a temporary dataset with random images and binary masks.

    Returns:
        dict with keys ``image_dir``, ``mask_dir``, ``image_paths``, ``mask_paths``.
    """
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    image_paths = []
    mask_paths = []

    rng = np.random.default_rng(seed=42)
    for i in range(4):
        img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        mask = rng.integers(0, 2, (128, 128), dtype=np.uint8) * 255

        img_path = image_dir / f"img_{i:03d}.png"
        msk_path = mask_dir / f"mask_{i:03d}.png"

        cv2.imwrite(str(img_path), img)
        cv2.imwrite(str(msk_path), mask)

        image_paths.append(img_path)
        mask_paths.append(msk_path)

    return {
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "image_paths": image_paths,
        "mask_paths": mask_paths,
    }


@pytest.fixture
def tmp_checkpoint(tmp_path: Path):
    """Save a minimal valid checkpoint and return its path."""
    model = nn.Sequential(nn.Conv2d(3, 1, 3, padding=1), nn.Sigmoid())
    ckpt = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "metrics": {"dice": 0.5, "iou": 0.3},
        "config": {},
        "history": [],
    }
    path = tmp_path / "checkpoint.pth"
    torch.save(ckpt, path)
    return path
