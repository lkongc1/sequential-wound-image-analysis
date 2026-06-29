"""Unit tests for src.models.factory registry pattern.

Run with: pytest tests/unit/test_model_factory.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch.nn as nn

# Setup project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import MODEL_REGISTRY, create_model, register_model


class TestRegistry:
    """Tests for the model registry and decorator."""

    def test_model_registry_is_dict(self):
        """MODEL_REGISTRY should be a dictionary."""
        assert isinstance(MODEL_REGISTRY, dict)

    def test_register_model_adds_to_registry(self):
        """@register_model should add a callable to MODEL_REGISTRY."""

        @register_model("dummy_test_model")
        def _dummy_factory() -> nn.Module:
            return nn.Identity()

        assert "dummy_test_model" in MODEL_REGISTRY
        assert MODEL_REGISTRY["dummy_test_model"] is _dummy_factory
        # Cleanup to avoid side effects in other tests
        del MODEL_REGISTRY["dummy_test_model"]

    def test_register_model_returns_original_callable(self):
        """The decorator should return the original function unchanged."""

        def _another_dummy() -> nn.Module:
            return nn.Identity()

        decorated = register_model("another_dummy")(_another_dummy)
        assert decorated is _another_dummy
        del MODEL_REGISTRY["another_dummy"]


class TestCreateModel:
    """Tests for create_model factory function."""

    def test_create_model_unknown_raises_valueerror(self):
        """create_model should raise ValueError for unknown model names."""
        with pytest.raises(ValueError) as exc_info:
            create_model("nonexistent_model")
        msg = str(exc_info.value)
        assert "nonexistent_model" in msg
        # Should list available models
        assert "Available models" in msg or "available models" in msg

    def test_create_unet_mini_returns_module(self):
        """create_model('unet_mini') should return an nn.Module."""
        model = create_model("unet_mini", in_channels=3, out_channels=1)
        assert isinstance(model, nn.Module)

    def test_create_attention_unet_returns_module(self):
        """create_model('attention_unet') should return an nn.Module."""
        model = create_model("attention_unet", in_channels=3, out_channels=1)
        assert isinstance(model, nn.Module)

    def test_create_nested_unet_returns_module(self):
        """create_model('nested_unet') should return an nn.Module."""
        model = create_model("nested_unet", in_channels=3, out_channels=1)
        assert isinstance(model, nn.Module)

    def test_create_unet_returns_module_if_smp_available(self):
        """create_model('unet') should return an nn.Module when SMP is installed."""
        try:
            import segmentation_models_pytorch as smp  # noqa: F401
        except ImportError:
            pytest.skip("segmentation_models_pytorch not installed")
        model = create_model("unet", encoder_name="resnet50", num_classes=1, pretrained=False)
        assert isinstance(model, nn.Module)

    def test_create_deeplabv3_returns_module_if_smp_available(self):
        """create_model('deeplabv3') should return an nn.Module when SMP is installed."""
        try:
            import segmentation_models_pytorch as smp  # noqa: F401
        except ImportError:
            pytest.skip("segmentation_models_pytorch not installed")
        model = create_model("deeplabv3", encoder_name="resnet50", num_classes=1, pretrained=False)
        assert isinstance(model, nn.Module)

    def test_create_manet_returns_module_if_smp_available(self):
        """create_model('manet') should return an nn.Module when SMP is installed."""
        try:
            import segmentation_models_pytorch as smp  # noqa: F401
        except ImportError:
            pytest.skip("segmentation_models_pytorch not installed")
        model = create_model("manet", encoder_name="resnet50", num_classes=1, pretrained=False)
        assert isinstance(model, nn.Module)

    def test_create_model_forwards_kwargs(self):
        """create_model should forward **kwargs to the registered factory."""
        model = create_model("unet_mini", in_channels=3, out_channels=2)
        # UNetMini ends with Conv2d(32, out_channels, 1)
        assert model.outc.out_channels == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
