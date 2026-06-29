"""Unit tests for MVP pipeline: config, model, and training step.

Run with: pytest tests/unit/test_mvp_pipeline.py -v
"""
from __future__ import annotations

import pytest
import torch
import sys
from pathlib import Path

# Setup project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainingConfig
from src.models.factory import create_model
from src.losses.bce_dice_loss import BCEDiceLoss


class TestTrainingConfig:
    """Test TrainingConfig dataclass following SRP principle."""

    def test_config_types(self):
        """Test 1: Verify TrainingConfig loads correctly with proper types."""
        config = TrainingConfig()

        # Verify types
        assert isinstance(config.cleaned_csv, Path)
        assert isinstance(config.checkpoint_dir, Path)
        assert isinstance(config.image_size, tuple)
        assert isinstance(config.batch_size, int)
        assert isinstance(config.max_epochs, int)
        assert isinstance(config.accelerator, str)
        assert isinstance(config.devices, int)
        assert isinstance(config.precision, str)

        # Verify precision is string (not int) for PyTorch Lightning
        assert config.precision == "16-mixed", (
            f"Precision should be '16-mixed', got {config.precision} (type: {type(config.precision).__name__})"
        )

    def test_config_default_values(self):
        """Verify default configuration values."""
        config = TrainingConfig()

        assert config.accelerator == "auto"
        assert config.devices == 1
        assert config.precision == "16-mixed"
        assert config.batch_size == 8
        assert config.max_epochs == 50
        assert config.learning_rate == 1e-4

    def test_config_paths_absolute(self):
        """Verify paths are resolved to absolute."""
        config = TrainingConfig()

        assert config.cleaned_csv.is_absolute()
        assert config.checkpoint_dir.is_absolute()


class TestModelInstantiation:
    """Test model creation following OCP principle."""

    def test_model_instantiation(self):
        """Test 2: Verify model creates and returns correct output shape."""
        model = create_model(name="unet", encoder_name="resnet50", num_classes=1)

        # Create dummy input: batch=2, channels=3, height=256, width=256
        dummy_input = torch.randn(2, 3, 256, 256)

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        # Verify output shape: (Batch, 1, H, W)
        assert output.shape == (2, 1, 256, 256), (
            f"Expected output shape (2, 1, 256, 256), got {output.shape}"
        )

    def test_model_output_finite(self):
        """Verify model output is finite (no NaN/Inf)."""
        model = create_model(name="unet", encoder_name="resnet50")
        dummy_input = torch.randn(1, 3, 128, 128)

        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        assert torch.isfinite(output).all(), "Model output contains NaN or Inf"


class TestGPUAvailability:
    """Test GPU detection and availability."""

    def test_gpu_availability(self):
        """Test 3: Verify GPU is available when CUDA is installed."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping GPU test")

        assert torch.cuda.is_available() is True
        assert torch.cuda.device_count() > 0

        device_name = torch.cuda.get_device_name(0)
        assert len(device_name) > 0
        assert "NVIDIA" in device_name or "GeForce" in device_name or "RTX" in device_name

    def test_gpu_memory_info(self):
        """Verify GPU memory can be queried."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping GPU memory test")

        mem_total = torch.cuda.get_device_properties(0).total_memory
        assert mem_total > 0
        assert mem_total > 1e9  # At least 1GB


class TestTrainingStepDryRun:
    """Test training step with dummy data (no actual training loop)."""

    def test_training_step_dry_run(self):
        """Test 4: Verify forward + backward pass with dummy data."""
        # Create model
        model_module = SegmentationModuleForTest(name="unet", encoder_name="resnet50")

        # Create dummy batch: (images, masks)
        batch_size = 2
        dummy_images = torch.randn(batch_size, 3, 256, 256)
        dummy_masks = (torch.randn(batch_size, 1, 256, 256) > 0).float()

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_module = model_module.to(device)
        dummy_images = dummy_images.to(device)
        dummy_masks = dummy_masks.to(device)

        # Forward pass
        model_module.train()
        outputs = model_module(dummy_images)

        # Compute loss
        loss_fn = BCEDiceLoss(bce_weight=0.3, dice_weight=0.7)
        loss = loss_fn(outputs, dummy_masks)

        # Verify loss is finite
        assert torch.isfinite(loss), "Loss contains NaN or Inf"
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

        # Backward pass
        loss.backward()

        # Verify gradients exist
        for name, param in model_module.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN or Inf"

    def test_validation_step_dry_run(self):
        """Verify validation step works with dummy data."""
        model_module = SegmentationModuleForTest(name="unet", encoder_name="resnet50")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_module = model_module.to(device)

        batch_size = 2
        dummy_images = torch.randn(batch_size, 3, 256, 256).to(device)
        dummy_masks = (torch.randn(batch_size, 1, 256, 256) > 0).float().to(device)

        model_module.eval()
        with torch.no_grad():
            outputs = model_module(dummy_images)
            loss_fn = BCEDiceLoss(bce_weight=0.3, dice_weight=0.7)
            loss = loss_fn(outputs, dummy_masks)

        assert torch.isfinite(loss), "Validation loss contains NaN or Inf"


# Inline module for testing (avoids importing full script)
class SegmentationModuleForTest(torch.nn.Module):
    """Simplified LightningModule for testing."""

    def __init__(self, name: str = "unet", encoder_name: str = "resnet50"):
        super().__init__()
        self.model = create_model(name=name, encoder_name=encoder_name)
        self.loss_fn = BCEDiceLoss(bce_weight=0.3, dice_weight=0.7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, masks = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        return loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
