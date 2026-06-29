"""Unit tests for src.metrics.clinical_metrics — f2_score and npv.

Run with: pytest tests/unit/test_clinical_metrics.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.clinical_metrics import f2_score, npv  # noqa: E402


class TestF2Score:
    """Tests for f2_score function."""

    def test_f2_score_known_values(self):
        """Known TP/FP/FN/TN — F2 matches expected value."""
        # TP=80, FP=10, FN=20, TN=890, beta=2
        # F2 = (1+4)*80 / ((1+4)*80 + 4*20 + 10) = 400 / (400+80+10) = 400/490 ≈ 0.8163
        result = f2_score(tp=80, fp=10, fn=20, tn=890)
        expected = 400.0 / 490.0
        assert result == pytest.approx(expected, rel=1e-4)

    def test_f2_score_favors_recall_over_precision(self):
        """F2 penalizes FN more than FP (FN-heavy case yields lower score)."""
        # Case A: FN=20, FP=5 — FN-heavy, F2 should be lower
        a = f2_score(tp=80, fp=5, fn=20, tn=895)
        # Case B: FN=5, FP=20 — FP-heavy, F2 should be higher
        b = f2_score(tp=80, fp=20, fn=5, tn=895)
        assert b > a, f"F2 should favor recall (FN penalty): A={a}, B={b}"

    def test_f2_score_zero_denominator(self):
        """All zeros → returns 0.0 gracefully."""
        result = f2_score(tp=0, fp=0, fn=0, tn=0)
        assert result == 0.0

    def test_f2_score_perfect(self):
        """Perfect prediction → F2 = 1.0."""
        result = f2_score(tp=100, fp=0, fn=0, tn=500)
        assert result == pytest.approx(1.0)

    def test_f2_score_no_predictions(self):
        """Model predicts nothing → F2 = 0.0 (TP + FP = 0)."""
        result = f2_score(tp=0, fp=0, fn=50, tn=950)
        assert result == 0.0

    def test_f2_score_custom_beta(self):
        """Custom beta parameter changes the weight."""
        # beta=1 is F1-score: (2*80) / (2*80+10+20) = 160/190 ≈ 0.8421
        f1 = f2_score(tp=80, fp=10, fn=20, tn=890, beta=1.0)
        expected_f1 = 160.0 / 190.0
        assert f1 == pytest.approx(expected_f1, rel=1e-4)


class TestNPV:
    """Tests for npv function."""

    def test_npv_known_values(self):
        """Known counts — NPV = TN/(TN+FN)."""
        result = npv(tn=890, fn=20)
        expected = 890.0 / 910.0
        assert result == pytest.approx(expected, rel=1e-4)

    def test_npv_zero_denominator(self):
        """Zero TN+FN → returns 0.0 gracefully."""
        result = npv(tn=0, fn=0)
        assert result == 0.0

    def test_npv_perfect_negative(self):
        """All negatives correctly identified → NPV = 1.0."""
        result = npv(tn=500, fn=0)
        assert result == pytest.approx(1.0)

    def test_npv_no_true_negatives(self):
        """No true negatives with FNs → NPV = 0.0."""
        result = npv(tn=0, fn=20)
        assert result == 0.0
