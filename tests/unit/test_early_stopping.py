"""Unit tests for src.training.early_stopping.

Run with: pytest tests/unit/test_early_stopping.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Setup project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.early_stopping import EarlyStopping


class TestEarlyStopping:
    """Tests for EarlyStopping helper."""

    def test_early_stopping_no_improvement_triggers(self):
        """Early stopping triggers after patience epochs without improvement."""
        es = EarlyStopping(patience=2, min_delta=0.01)
        es(0.5)
        assert not es.early_stop
        es(0.5)
        assert not es.early_stop  # no improvement, counter 1
        es(0.5)
        assert es.early_stop  # counter reached patience

    def test_early_stopping_improvement_resets(self):
        """Improvement resets the counter."""
        es = EarlyStopping(patience=2, min_delta=0.01)
        es(0.5)
        es(0.6)
        assert not es.early_stop  # improved (higher is better)
        es(0.6)
        assert not es.early_stop  # counter 1
        es(0.6)
        assert es.early_stop  # counter reached patience

    def test_early_stopping_min_delta(self):
        """Changes below min_delta do not count as improvement."""
        es = EarlyStopping(patience=1, min_delta=0.1)
        es(0.5)
        es(0.55)
        assert es.early_stop  # delta 0.05 < 0.1

    def test_early_stopping_best_score_tracked(self):
        """Best score is updated and retained."""
        es = EarlyStopping(patience=1, min_delta=0.0)
        es(0.5)
        assert es.best_score == 0.5
        es(0.7)
        assert es.best_score == 0.7
        es(0.7)
        assert es.early_stop
        assert es.best_score == 0.7

    def test_early_stopping_mode_min(self):
        """EarlyStopping works in 'min' mode (lower is better)."""
        es = EarlyStopping(patience=2, min_delta=0.01, mode="min")
        es(0.5)
        es(0.5)
        assert not es.early_stop
        es(0.5)
        assert es.early_stop

    def test_early_stopping_min_mode_improvement(self):
        """Improvement in min mode resets counter."""
        es = EarlyStopping(patience=2, min_delta=0.01, mode="min")
        es(0.5)
        es(0.4)
        assert not es.early_stop
        es(0.4)
        assert not es.early_stop
        es(0.4)
        assert es.early_stop


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
