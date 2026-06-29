"""Integration tests for inference predictor.

Run with: pytest tests/integration/test_inference.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import Predictor


class TestPredictorPredict:
    """Tests for Predictor.predict() behaviour."""

    def test_predict_runs_on_cpu(self, dummy_model):
        """predict() should return a sigmoid-activated tensor on CPU."""
        predictor = Predictor(dummy_model, device="cpu")
        image = torch.randn(1, 3, 64, 64)

        output = predictor.predict(image)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 1, 64, 64)
        # Already sigmoid-activated by Predictor
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_predict_batch_runs_on_cpu(self, dummy_model):
        """predict_batch() should handle multiple images."""
        predictor = Predictor(dummy_model, device="cpu")
        images = torch.randn(4, 3, 64, 64)

        outputs = predictor.predict_batch(images)

        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (4, 1, 64, 64)
        assert outputs.min() >= 0.0
        assert outputs.max() <= 1.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_predict_uses_cuda_when_available(self, dummy_model):
        """If CUDA is present, predictor should default to cuda."""
        predictor = Predictor(dummy_model)  # device defaults to cuda if available
        assert predictor.device == "cuda"
        image = torch.randn(1, 3, 64, 64)
        output = predictor.predict(image)
        assert output.device.type == "cuda"

    def test_predict_explicit_cpu(self, dummy_model):
        """Explicit device='cpu' should keep tensors on CPU."""
        predictor = Predictor(dummy_model, device="cpu")
        image = torch.randn(1, 3, 64, 64)
        output = predictor.predict(image)
        assert output.device.type == "cpu"


class TestPredictorPostprocessing:
    """Tests for thresholding / postprocessing of predictor outputs."""

    def test_sigmoid_output_can_be_thresholded(self, dummy_model):
        """Sigmoid output can be binarised with a simple threshold."""
        predictor = Predictor(dummy_model, device="cpu")
        image = torch.randn(1, 3, 64, 64)

        prob = predictor.predict(image)
        threshold = 0.5
        binary = (prob > threshold).float()

        assert torch.all((binary == 0) | (binary == 1))
        assert binary.shape == prob.shape

    def test_extreme_thresholds(self, dummy_model):
        """Thresholds of 0.0 and 1.0 should yield all-ones and all-zeros."""
        predictor = Predictor(dummy_model, device="cpu")
        image = torch.randn(1, 3, 64, 64)

        prob = predictor.predict(image)
        # Sigmoid never actually reaches exactly 0 or 1 for finite inputs,
        # but with a dummy conv+sigmoid the outputs are strictly in (0,1).
        all_ones = (prob > 0.0).float()
        all_zeros = (prob >= 1.0).float()

        assert torch.all(all_ones == 1)
        assert torch.all(all_zeros == 0)


class TestPredictorCheckpoint:
    """Tests for checkpoint loading integration."""

    def test_load_checkpoint_restores_weights(self, dummy_model, tmp_checkpoint):
        """load_checkpoint should restore model weights from a .pth file."""
        predictor = Predictor(dummy_model, device="cpu")

        # Mutate current weights
        with torch.no_grad():
            for p in predictor.model.parameters():
                p.fill_(0.0)

        predictor.load_checkpoint(str(tmp_checkpoint))

        any_nonzero = any((p != 0).any().item() for p in predictor.model.parameters())
        assert any_nonzero
