"""Unit tests for src.training.checkpoint_manager.

Run with: pytest tests/unit/test_checkpoint_manager.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Setup project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.checkpoint_manager import CheckpointManager


def _dummy_model() -> nn.Module:
    return nn.Conv2d(3, 1, 1)


def _dummy_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


class TestCheckpointManagerInit:
    """Tests for CheckpointManager initialization."""

    def test_init_creates_output_dir(self, tmp_path: Path):
        """Constructor creates the output directory if it doesn't exist."""
        output_dir = tmp_path / "checkpoints"
        assert not output_dir.exists()
        CheckpointManager(output_dir=output_dir)
        assert output_dir.exists()

    def test_init_defaults(self, tmp_path: Path):
        """Default monitor and mode are set correctly."""
        cm = CheckpointManager(output_dir=tmp_path)
        assert cm.monitor == "val_dice"
        assert cm.mode == "max"

    def test_init_custom_monitor_and_mode(self, tmp_path: Path):
        """Custom monitor and mode are accepted."""
        cm = CheckpointManager(output_dir=tmp_path, monitor="val_loss", mode="min")
        assert cm.monitor == "val_loss"
        assert cm.mode == "min"


class TestCheckpointManagerIsBetter:
    """Tests for CheckpointManager.is_better."""

    def test_is_better_max(self, tmp_path: Path):
        """Higher value is better in max mode."""
        cm = CheckpointManager(output_dir=tmp_path, mode="max")
        assert cm.is_better(0.6, 0.5)
        assert not cm.is_better(0.5, 0.6)
        assert not cm.is_better(0.5, 0.5)

    def test_is_better_min(self, tmp_path: Path):
        """Lower value is better in min mode."""
        cm = CheckpointManager(output_dir=tmp_path, mode="min")
        assert cm.is_better(0.4, 0.5)
        assert not cm.is_better(0.5, 0.4)
        assert not cm.is_better(0.5, 0.5)

    def test_is_better_no_best(self, tmp_path: Path):
        """Any value is better when there is no best yet."""
        cm = CheckpointManager(output_dir=tmp_path, mode="max")
        assert cm.is_better(0.5, None)
        cm2 = CheckpointManager(output_dir=tmp_path, mode="min")
        assert cm2.is_better(0.5, None)


class TestCheckpointManagerSave:
    """Tests for CheckpointManager.save."""

    def test_save_creates_file(self, tmp_path: Path):
        """save writes a .pth file with state_dicts and metadata."""
        cm = CheckpointManager(output_dir=tmp_path)
        model = _dummy_model()
        optimizer = _dummy_optimizer(model)

        cm.save(model=model, optimizer=optimizer, epoch=3, metrics={"val_dice": 0.75})

        files = list(tmp_path.glob("*.pth"))
        assert len(files) == 1
        ckpt = torch.load(files[0], map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "epoch" in ckpt
        assert ckpt["epoch"] == 3
        assert "metrics" in ckpt
        assert ckpt["metrics"]["val_dice"] == 0.75

    def test_save_overwrites_on_better_metric(self, tmp_path: Path):
        """Only the best checkpoint is kept when save is called multiple times."""
        cm = CheckpointManager(output_dir=tmp_path, monitor="val_dice", mode="max")
        model = _dummy_model()
        optimizer = _dummy_optimizer(model)

        cm.save(model=model, optimizer=optimizer, epoch=1, metrics={"val_dice": 0.5})
        first_file = list(tmp_path.glob("*.pth"))[0]
        first_mtime = first_file.stat().st_mtime

        # Same metric — should not overwrite
        cm.save(model=model, optimizer=optimizer, epoch=2, metrics={"val_dice": 0.5})
        assert list(tmp_path.glob("*.pth"))[0].stat().st_mtime == first_mtime

        # Better metric — should overwrite
        cm.save(model=model, optimizer=optimizer, epoch=3, metrics={"val_dice": 0.8})
        files = list(tmp_path.glob("*.pth"))
        assert len(files) == 1
        ckpt = torch.load(files[0], map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 3
        assert ckpt["metrics"]["val_dice"] == 0.8

    def test_save_min_mode(self, tmp_path: Path):
        """save respects min mode for metric comparison."""
        cm = CheckpointManager(output_dir=tmp_path, monitor="val_loss", mode="min")
        model = _dummy_model()
        optimizer = _dummy_optimizer(model)

        cm.save(model=model, optimizer=optimizer, epoch=1, metrics={"val_loss": 0.5})
        cm.save(model=model, optimizer=optimizer, epoch=2, metrics={"val_loss": 0.3})

        files = list(tmp_path.glob("*.pth"))
        assert len(files) == 1
        ckpt = torch.load(files[0], map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 2
        assert ckpt["metrics"]["val_loss"] == 0.3


class TestCheckpointManagerLoad:
    """Tests for CheckpointManager.load."""

    def test_load_returns_dict(self, tmp_path: Path):
        """load returns a dict with state dicts and metadata."""
        cm = CheckpointManager(output_dir=tmp_path)
        model = _dummy_model()
        optimizer = _dummy_optimizer(model)

        cm.save(model=model, optimizer=optimizer, epoch=5, metrics={"val_dice": 0.9})
        ckpt_path = list(tmp_path.glob("*.pth"))[0]

        result = cm.load(ckpt_path)
        assert isinstance(result, dict)
        assert "model_state_dict" in result
        assert "optimizer_state_dict" in result
        assert "epoch" in result
        assert result["epoch"] == 5
        assert "metrics" in result

    def test_load_restores_state(self, tmp_path: Path):
        """load restores model and optimizer state correctly."""
        cm = CheckpointManager(output_dir=tmp_path)
        model = _dummy_model()
        optimizer = _dummy_optimizer(model)
        # Step once to populate optimizer state
        dummy_input = torch.randn(1, 3, 8, 8)
        loss = model(dummy_input).sum()
        loss.backward()
        optimizer.step()

        original_param = next(model.parameters()).clone()
        cm.save(model=model, optimizer=optimizer, epoch=1, metrics={"val_dice": 0.5})
        ckpt_path = list(tmp_path.glob("*.pth"))[0]

        # Mutate model
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()

        result = cm.load(ckpt_path)
        model.load_state_dict(result["model_state_dict"])
        restored_param = next(model.parameters())
        assert torch.allclose(restored_param, original_param)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
