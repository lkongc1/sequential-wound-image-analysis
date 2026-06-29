"""Integration test: end-to-end training pipeline.

Runs the complete flow from data splitting through one training epoch
using only CPU and tiny synthetic data so the test stays under 5 s.

Run with: pytest tests/integration/test_pipeline.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing.splitter import DataSplitter
from src.datasets.wound_dataset import WoundDataset, get_default_transforms
from src.training.trainer import Trainer


def test_end_to_end_pipeline(synthetic_dataset):
    """Full pipeline: split -> dataset -> loaders -> train 1 epoch -> history."""
    # ------------------------------------------------------------------
    # 1. Patient-aware split with DataSplitter
    # ------------------------------------------------------------------
    image_paths = synthetic_dataset["image_paths"]
    mask_paths = synthetic_dataset["mask_paths"]

    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3", "p4"],
            "image_path": [str(p) for p in image_paths],
            "mask_path": [str(p) for p in mask_paths],
        }
    )

    splitter = DataSplitter(val_size=0.25, test_size=0.25, random_state=42)
    df_split = splitter.split_all(df)

    assert "split" in df_split.columns
    assert set(df_split["split"].unique()).issubset({"train", "val", "test"})

    # ------------------------------------------------------------------
    # 2. Create WoundDatasets per split (lightweight 64x64 transforms)
    # ------------------------------------------------------------------
    transform = get_default_transforms(image_size=(64, 64))

    def _make_dataset(subset_df):
        return WoundDataset(
            image_paths=[Path(p) for p in subset_df["image_path"].tolist()],
            mask_paths=[Path(p) for p in subset_df["mask_path"].tolist()],
            transform=transform,
        )

    train_df = df_split[df_split["split"] == "train"]
    val_df = df_split[df_split["split"] == "val"]

    train_ds = _make_dataset(train_df)
    val_ds = _make_dataset(val_df)

    assert len(train_ds) > 0
    assert len(val_ds) > 0

    # ------------------------------------------------------------------
    # 3. Create DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    # ------------------------------------------------------------------
    # 4. Instantiate Trainer with dummy model and minimal deps
    # ------------------------------------------------------------------
    dummy_model = nn.Sequential(
        nn.Conv2d(3, 1, kernel_size=3, padding=1),
        nn.Sigmoid(),
    )

    def optimizer_factory(params):
        return torch.optim.SGD(params, lr=1e-3)

    # Use BCELoss because dummy_model already has Sigmoid.
    # If we used BCEWithLogitsLoss we would need to remove Sigmoid.
    trainer = Trainer(
        model=dummy_model,
        criterion=nn.BCELoss(),
        optimizer_factory=optimizer_factory,
        scheduler_factory=None,
        early_stopping=None,
        checkpoint_manager=None,
        device="cpu",
    )

    # ------------------------------------------------------------------
    # 5. Run fit() for 1 epoch
    # ------------------------------------------------------------------
    history = trainer.fit(train_loader, val_loader, epochs=1)

    # ------------------------------------------------------------------
    # 6. Verify history
    # ------------------------------------------------------------------
    assert len(history) == 1
    assert "train_loss" in history[0]
    assert "val_loss" in history[0]
    assert "val_dice" in history[0]
    assert "val_iou" in history[0]
    assert history[0]["epoch"] == 1

    # Ensure no NaN/Inf crept in
    for key in ("train_loss", "val_loss", "val_dice", "val_iou"):
        assert torch.isfinite(torch.tensor(history[0][key]))
