"""Unit tests for src.training.trainer.

Run with: pytest tests/unit/test_trainer.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Setup project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.checkpoint_manager import CheckpointManager
from src.training.early_stopping import EarlyStopping
from src.training.trainer import Trainer


def _dummy_loaders(batch_size: int = 2, num_samples: int = 4, img_size: int = 32):
    """Create tiny dummy DataLoaders for fast testing."""
    x = torch.randn(num_samples, 3, img_size, img_size)
    y = (torch.randn(num_samples, 1, img_size, img_size) > 0).float()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, loader  # train and val same


def _default_criterion() -> nn.Module:
    return nn.BCEWithLogitsLoss()


def _default_optimizer_factory(params):
    return torch.optim.AdamW(params, lr=1e-3)


class TestTrainerInit:
    """Tests for Trainer initialization."""

    def test_trainer_has_expected_attributes(self):
        """Trainer initializes with required attributes."""
        model = nn.Conv2d(3, 1, 1)
        device = torch.device("cpu")
        trainer = Trainer(
            model=model,
            criterion=_default_criterion(),
            optimizer_factory=_default_optimizer_factory,
            device=device,
        )

        assert trainer.model is model
        assert trainer.device == device
        assert isinstance(trainer.history, list)
        assert hasattr(trainer, "scaler")

    def test_trainer_with_config(self):
        """Trainer accepts optional config dict."""
        model = nn.Conv2d(3, 1, 1)
        config = {"lr": 1e-3}
        trainer = Trainer(
            model=model,
            criterion=_default_criterion(),
            optimizer_factory=_default_optimizer_factory,
            device="cpu",
            config=config,
        )
        assert trainer.config == config


class TestTrainerTrainEpoch:
    """Tests for Trainer.train_epoch."""

    def test_train_epoch_runs_and_returns_loss(self):
        """A training epoch runs and returns a finite scalar loss."""
        model = nn.Conv2d(3, 1, 1)
        trainer = Trainer(
            model=model,
            criterion=_default_criterion(),
            optimizer_factory=_default_optimizer_factory,
            device="cpu",
        )
        train_loader, _ = _dummy_loaders()

        loss = trainer.train_epoch(train_loader)
        assert isinstance(loss, float)
        assert loss >= 0.0


class TestTrainerValidate:
    """Tests for Trainer.validate."""

    def test_validate_returns_metrics(self):
        """Validation returns a dict with loss, dice, and iou."""
        model = nn.Conv2d(3, 1, 1)
        trainer = Trainer(
            model=model,
            criterion=_default_criterion(),
            optimizer_factory=_default_optimizer_factory,
            device="cpu",
        )
        val_loader, _ = _dummy_loaders()

        metrics = trainer.validate(val_loader)
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "dice" in metrics
        assert "iou" in metrics
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


class TestTrainerFit:
    """Tests for Trainer.fit."""

    def test_fit_runs_multiple_epochs(self):
        """fit runs for the requested number of epochs."""
        model = nn.Conv2d(3, 1, 1)
        trainer = Trainer(
            model=model,
            criterion=_default_criterion(),
            optimizer_factory=_default_optimizer_factory,
            device="cpu",
        )
        train_loader, val_loader = _dummy_loaders()

        trainer.fit(train_loader, val_loader, epochs=2)
        assert len(trainer.history) == 2
        assert "train_loss" in trainer.history[0]
        assert "val_dice" in trainer.history[0]

    def test_fit_saves_best_checkpoint(self, tmp_path: Path):
        """fit saves the best checkpoint via CheckpointManager."""
        model = nn.Conv2d(3, 1, 1)
        ckpt_mgr = CheckpointManager(output_dir=tmp_path, monitor="val_dice", mode="max")
        trainer = Trainer(
            model=model,
            criterion=_default_criterion(),
            optimizer_factory=_default_optimizer_factory,
            checkpoint_manager=ckpt_mgr,
            device="cpu",
        )
        train_loader, val_loader = _dummy_loaders()

        trainer.fit(train_loader, val_loader, epochs=2)
        checkpoints = list(tmp_path.glob("*.pth"))
        assert len(checkpoints) >= 1

    def test_fit_respects_early_stopping(self):
        """fit stops early when EarlyStopping triggers."""
        model = nn.Conv2d(3, 1, 1)
        early_stopping = EarlyStopping(patience=1, min_delta=1e-4)
        trainer = Trainer(
            model=model,
            criterion=_default_criterion(),
            optimizer_factory=_default_optimizer_factory,
            early_stopping=early_stopping,
            device="cpu",
        )
        train_loader, val_loader = _dummy_loaders()

        trainer.fit(train_loader, val_loader, epochs=10)
        # Should stop before 10 epochs due to no improvement
        assert len(trainer.history) < 10


class TestTrainerScheduler:
    """Tests for scheduler integration."""

    def test_scheduler_receives_val_metric(self):
        """fit calls scheduler.step(val_dice) when a scheduler is injected."""
        model = nn.Conv2d(3, 1, 1)
        scheduler_calls = []

        class DummyScheduler:
            def __init__(self, optimizer):
                self.optimizer = optimizer

            def step(self, metric):
                scheduler_calls.append(metric)

        def _scheduler_factory(opt):
            return DummyScheduler(opt)

        criterion = _default_criterion()
        optimizer_factory = _default_optimizer_factory
        scheduler_factory = _scheduler_factory
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            device="cpu",
        )
        train_loader, val_loader = _dummy_loaders()

        trainer.fit(train_loader, val_loader, epochs=2)
        assert len(scheduler_calls) == 2
        assert all(isinstance(m, float) for m in scheduler_calls)


class TestTrainerCheckpoint:
    """Tests for save_checkpoint and load_checkpoint."""

    def test_save_checkpoint_creates_file(self, tmp_path: Path):
        """save_checkpoint writes a .pth file with metadata."""
        model = nn.Conv2d(3, 1, 1)
        trainer = Trainer(
            model=model,
            criterion=_default_criterion(),
            optimizer_factory=_default_optimizer_factory,
            device="cpu",
        )
        ckpt_path = tmp_path / "test_ckpt.pth"
        trainer.save_checkpoint(epoch=1, metrics={"dice": 0.5}, path=ckpt_path)

        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "epoch" in ckpt
        assert ckpt["epoch"] == 1

    def test_load_checkpoint_restores_state(self, tmp_path: Path):
        """load_checkpoint restores model weights."""
        model = nn.Conv2d(3, 1, 1)
        trainer = Trainer(
            model=model,
            criterion=_default_criterion(),
            optimizer_factory=_default_optimizer_factory,
            device="cpu",
        )
        ckpt_path = tmp_path / "test_ckpt.pth"
        trainer.save_checkpoint(epoch=2, metrics={"dice": 0.7}, path=ckpt_path)

        # Mutate model weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0.0)

        trainer.load_checkpoint(ckpt_path)
        # At least one parameter should be non-zero after loading
        any_nonzero = any((p != 0).any().item() for p in model.parameters())
        assert any_nonzero


class TestTrainerEvaluate:
    """Tests for Trainer.evaluate."""

    def test_evaluate_returns_metrics(self):
        """evaluate returns aggregated metrics on test set."""
        model = nn.Conv2d(3, 1, 1)
        trainer = Trainer(
            model=model,
            criterion=_default_criterion(),
            optimizer_factory=_default_optimizer_factory,
            device="cpu",
        )
        test_loader, _ = _dummy_loaders()

        metrics = trainer.evaluate(test_loader)
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "dice" in metrics
        assert "iou" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
