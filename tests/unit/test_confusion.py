"""Unit tests for src.metrics.confusion — ConfusionMatrix dataclass.

Run with: pytest tests/unit/test_confusion.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.confusion import ConfusionMatrix  # noqa: E402


class TestConfusionMatrixInit:
    """Tests for ConfusionMatrix initialization."""

    def test_defaults_to_zero(self):
        """All counts start at zero."""
        cm = ConfusionMatrix()
        assert cm.tp == 0
        assert cm.fp == 0
        assert cm.fn == 0
        assert cm.tn == 0

    def test_custom_initial_values(self):
        """Custom initial values accepted."""
        cm = ConfusionMatrix(tp=10, fp=5, fn=2, tn=100)
        assert cm.tp == 10
        assert cm.fp == 5
        assert cm.fn == 2
        assert cm.tn == 100


class TestConfusionMatrixAccumulate:
    """Tests for ConfusionMatrix.accumulate() method."""

    def test_accumulate_all_ones(self):
        """All-ones predictions and targets yield only TP."""
        cm = ConfusionMatrix()
        preds = torch.ones(2, 1, 8, 8)
        targets = torch.ones(2, 1, 8, 8)
        cm.accumulate(preds, targets, threshold=0.5)
        assert cm.tp == 2 * 64  # 2 images x 64 pixels
        assert cm.fp == 0
        assert cm.fn == 0
        assert cm.tn == 0

    def test_accumulate_all_zeros(self):
        """All-zero predictions, all-zero targets yield only TN."""
        cm = ConfusionMatrix()
        preds = torch.zeros(2, 1, 8, 8)
        targets = torch.zeros(2, 1, 8, 8)
        cm.accumulate(preds, targets, threshold=0.5)
        assert cm.tp == 0
        assert cm.fp == 0
        assert cm.fn == 0
        assert cm.tn == 2 * 64

    def test_accumulate_false_positives(self):
        """Predict ones on zero targets → FP."""
        cm = ConfusionMatrix()
        preds = torch.ones(1, 1, 4, 4)
        targets = torch.zeros(1, 1, 4, 4)
        cm.accumulate(preds, targets, threshold=0.5)
        assert cm.tp == 0
        assert cm.fp == 16
        assert cm.fn == 0
        assert cm.tn == 0

    def test_accumulate_false_negatives(self):
        """Predict zeros on one targets → FN."""
        cm = ConfusionMatrix()
        preds = torch.zeros(1, 1, 4, 4)
        targets = torch.ones(1, 1, 4, 4)
        cm.accumulate(preds, targets, threshold=0.5)
        assert cm.tp == 0
        assert cm.fp == 0
        assert cm.fn == 16
        assert cm.tn == 0

    def test_accumulate_multiple_batches(self):
        """Two batches accumulate cumulatively."""
        cm = ConfusionMatrix()
        # Batch 1: all TP
        cm.accumulate(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4), threshold=0.5)
        assert cm.tp == 16
        # Batch 2: all TN
        cm.accumulate(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), threshold=0.5)
        assert cm.tp == 16
        assert cm.tn == 16
        assert cm.fp == 0
        assert cm.fn == 0

    def test_accumulate_custom_threshold(self):
        """Threshold affects binarization."""
        cm = ConfusionMatrix()
        # Predictions of 0.3 with threshold 0.5 → all zeros (below threshold)
        preds = torch.full((1, 1, 4, 4), 0.3)
        targets = torch.ones(1, 1, 4, 4)
        cm.accumulate(preds, targets, threshold=0.5)
        assert cm.tp == 0
        assert cm.fn == 16  # all missed

        # Same predictions with threshold 0.2 → all ones
        cm2 = ConfusionMatrix()
        cm2.accumulate(preds, targets, threshold=0.2)
        assert cm2.tp == 16
        assert cm2.fn == 0

    def test_accumulate_mixed_batch(self):
        """Half TP, half FP in a single batch."""
        cm = ConfusionMatrix()
        preds = torch.tensor([[[[1.0, 1.0],
                                [0.0, 0.0]]]])  # shape (1,1,2,2)
        targets = torch.tensor([[[[1.0, 0.0],
                                  [0.0, 0.0]]]])
        cm.accumulate(preds, targets, threshold=0.5)
        # pred:   [[1,1],[0,0]]  target: [[1,0],[0,0]]
        # (0,0): pred=1, targ=1 → TP
        # (0,1): pred=1, targ=0 → FP
        # (1,0): pred=0, targ=0 → TN
        # (1,1): pred=0, targ=0 → TN
        assert cm.tp == 1
        assert cm.fp == 1
        assert cm.fn == 0
        assert cm.tn == 2


class TestConfusionMatrixDeriveMetrics:
    """Tests for ConfusionMatrix.derive_metrics() method."""

    def test_derive_known_values(self):
        """Known confusion matrix — derived metrics match spec."""
        cm = ConfusionMatrix(tp=80, fp=10, fn=20, tn=890)
        metrics = cm.derive_metrics()
        assert metrics["sensitivity"] == pytest.approx(0.80, rel=1e-2)
        assert metrics["specificity"] == pytest.approx(890 / 900, rel=1e-3)
        assert metrics["precision"] == pytest.approx(80 / 90, rel=1e-2)
        assert metrics["dice"] == pytest.approx(2 * 80 / (2 * 80 + 10 + 20), rel=1e-2)
        assert metrics["iou"] == pytest.approx(80 / (80 + 10 + 20), rel=1e-2)
        assert metrics["accuracy"] == pytest.approx((80 + 890) / 1000, rel=1e-3)
        assert "npv" in metrics
        assert "f2" in metrics

    def test_derive_all_zeros(self):
        """All-zero counts → all metrics return 0.0."""
        cm = ConfusionMatrix()
        metrics = cm.derive_metrics()
        for key in metrics:
            assert metrics[key] == 0.0, f"Metric '{key}' should be 0.0 for empty matrix, got {metrics[key]}"

    def test_derive_perfect_prediction(self):
        """Perfect prediction → all metrics at 1.0."""
        cm = ConfusionMatrix(tp=100, fp=0, fn=0, tn=500)
        metrics = cm.derive_metrics()
        assert metrics["sensitivity"] == pytest.approx(1.0)
        assert metrics["specificity"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["dice"] == pytest.approx(1.0)
        assert metrics["iou"] == pytest.approx(1.0)
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_derive_returns_all_expected_keys(self):
        """All expected metrics are present in the returned dict."""
        cm = ConfusionMatrix(tp=1, fp=0, fn=0, tn=1)
        metrics = cm.derive_metrics()
        expected_keys = {
            "sensitivity", "specificity", "precision", "npv",
            "f2", "dice", "iou", "accuracy",
        }
        assert set(metrics.keys()) == expected_keys


class TestConfusionMatrixToNumpy:
    """Tests for ConfusionMatrix.to_numpy() method."""

    def test_to_numpy_returns_tuple_of_arrays(self):
        """Returns (absolute_2x2, normalized_2x2)."""
        cm = ConfusionMatrix(tp=80, fp=10, fn=20, tn=890)
        abs_mat, norm_mat = cm.to_numpy()
        assert isinstance(abs_mat, np.ndarray)
        assert isinstance(norm_mat, np.ndarray)
        assert abs_mat.shape == (2, 2)
        assert norm_mat.shape == (2, 2)

    def test_to_numpy_absolute_values(self):
        """Absolute matrix contains raw counts."""
        cm = ConfusionMatrix(tp=80, fp=10, fn=20, tn=890)
        abs_mat, _ = cm.to_numpy()
        # Convention: row=actual, col=predicted
        # [[TN, FP], [FN, TP]]
        assert abs_mat[0, 0] == 890  # TN
        assert abs_mat[0, 1] == 10   # FP
        assert abs_mat[1, 0] == 20   # FN
        assert abs_mat[1, 1] == 80   # TP

    def test_to_numpy_normalized_sums_to_one(self):
        """Normalized matrix total sums to 1.0."""
        cm = ConfusionMatrix(tp=80, fp=10, fn=20, tn=890)
        _, norm_mat = cm.to_numpy()
        # Global normalization: all entries divided by total count
        assert pytest.approx(norm_mat.sum()) == 1.0

    def test_to_numpy_all_zeros(self):
        """Zero counts → zero arrays."""
        cm = ConfusionMatrix()
        abs_mat, norm_mat = cm.to_numpy()
        assert abs_mat.sum() == 0
        assert norm_mat.sum() == 0  # no division since total is 0
