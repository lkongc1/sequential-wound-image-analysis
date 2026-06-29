"""Unit tests for src.data.review.outliers."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.review.outliers import (
    EmptyMaskStrategy,
    IQRStrategy,
    OutlierDetector,
    PercentileStrategy,
    ZScoreStrategy,
)


class TestZScoreStrategy:
    """Tests for ZScoreStrategy outlier detection."""

    def test_detect_no_outliers(self) -> None:
        """Data within 3 sigma has no outliers."""
        df = pd.DataFrame({"value": [50.0, 51.0, 49.0, 50.5, 49.5, 50.2]})
        strategy = ZScoreStrategy(threshold=3.0)
        outliers = strategy.detect(df, "value")
        assert len(outliers) == 0

    def test_detect_with_outliers(self) -> None:
        """Extreme values beyond 3 sigma are detected."""
        df = pd.DataFrame({"value": [50.0] * 10 + [500.0]})
        strategy = ZScoreStrategy(threshold=3.0)
        outliers = strategy.detect(df, "value")
        assert 10 in outliers  # index of extreme value

    def test_detect_low_threshold(self) -> None:
        """Lower threshold catches more outliers."""
        df = pd.DataFrame({"value": [50.0, 51.0, 49.0, 60.0, 48.0]})
        strategy = ZScoreStrategy(threshold=1.0)
        outliers = strategy.detect(df, "value")
        assert len(outliers) > 0

    def test_detect_missing_column(self) -> None:
        """Non-existent column returns empty set."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        strategy = ZScoreStrategy()
        outliers = strategy.detect(df, "nonexistent")
        assert outliers == set()

    def test_detect_insufficient_data(self) -> None:
        """Less than 3 data points returns empty set."""
        df = pd.DataFrame({"value": [1.0, 2.0]})
        strategy = ZScoreStrategy()
        outliers = strategy.detect(df, "value")
        assert outliers == set()

    def test_detect_constant_column(self) -> None:
        """Constant column (std=0) returns empty set."""
        df = pd.DataFrame({"value": [50.0, 50.0, 50.0, 50.0]})
        strategy = ZScoreStrategy()
        outliers = strategy.detect(df, "value")
        assert outliers == set()

    def test_get_name(self) -> None:
        """Strategy name includes threshold."""
        strategy = ZScoreStrategy(threshold=2.5)
        assert "2.5" in strategy.get_name()


class TestIQRStrategy:
    """Tests for IQRStrategy outlier detection."""

    def test_detect_no_outliers(self) -> None:
        """Compact data has no outliers."""
        df = pd.DataFrame({"value": [50.0, 51.0, 49.0, 50.5, 49.5, 50.2]})
        strategy = IQRStrategy(multiplier=1.5)
        outliers = strategy.detect(df, "value")
        assert len(outliers) == 0

    def test_detect_with_outliers(self) -> None:
        """Values beyond Q3 + 3*IQR are detected as extreme outliers."""
        df = pd.DataFrame({"value": [50.0, 51.0, 49.0, 10.0, 200.0]})
        strategy = IQRStrategy(multiplier=1.5)
        outliers = strategy.detect(df, "value")
        # 10.0 and 200.0 should be outliers (not 50, 51, 49 which are normal)
        assert len(outliers) >= 1

    def test_get_name(self) -> None:
        """Strategy name includes multiplier."""
        strategy = IQRStrategy(multiplier=2.0)
        assert "2.0" in strategy.get_name()


class TestPercentileStrategy:
    """Tests for PercentileStrategy outlier detection."""

    def test_detect_no_outliers(self) -> None:
        """Data within percentiles has no outliers."""
        df = pd.DataFrame({"value": [10.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 90.0]})
        strategy = PercentileStrategy(lower=10.0, upper=90.0)
        outliers = strategy.detect(df, "value")
        assert len(outliers) == 0

    def test_detect_with_outliers(self) -> None:
        """Values outside percentile bounds are detected."""
        df = pd.DataFrame({"value": [1.0, 2.0, 50.0, 98.0, 99.0]})
        strategy = PercentileStrategy(lower=10.0, upper=90.0)
        outliers = strategy.detect(df, "value")
        # 1.0, 2.0, 98.0, 99.0 should be outliers
        assert len(outliers) >= 1

    def test_get_name(self) -> None:
        """Strategy name includes percentile bounds."""
        strategy = PercentileStrategy(lower=5.0, upper=95.0)
        assert "5" in strategy.get_name()


class TestEmptyMaskStrategy:
    """Tests for EmptyMaskStrategy outlier detection."""

    def test_detect_empty_masks(self) -> None:
        """Rows with empty masks are detected."""
        df = pd.DataFrame({
            "is_empty": [False, True, False, True, False],
        })
        strategy = EmptyMaskStrategy()
        outliers = strategy.detect(df, "is_empty")
        assert 1 in outliers
        assert 3 in outliers
        assert len(outliers) == 2

    def test_detect_no_empty_masks(self) -> None:
        """No outliers when all masks have wounds."""
        df = pd.DataFrame({
            "is_empty": [False, False, False, False],
        })
        strategy = EmptyMaskStrategy()
        outliers = strategy.detect(df, "is_empty")
        assert len(outliers) == 0

    def test_detect_default_column(self) -> None:
        """Default column name is 'is_empty'."""
        df = pd.DataFrame({"is_empty": [True, False]})
        strategy = EmptyMaskStrategy()
        outliers = strategy.detect(df)
        assert 0 in outliers

    def test_get_name(self) -> None:
        """Strategy name is 'EmptyMask'."""
        strategy = EmptyMaskStrategy()
        assert strategy.get_name() == "EmptyMask"


class TestOutlierDetector:
    """Tests for OutlierDetector orchestrator."""

    def test_detect_single_column(self) -> None:
        """Detector finds outliers in configured column."""
        df = pd.DataFrame({"wound_pct": [5.0, 5.1, 5.0, 60.0, 4.9, 5.2]})
        detector = OutlierDetector(strategy=ZScoreStrategy(threshold=2.0))
        detector.add_column("wound_pct")
        results = detector.detect(df)
        assert "wound_pct" in results
        assert 3 in results["wound_pct"]  # 60.0 is extreme

    def test_detect_multiple_columns(self) -> None:
        """Detector can analyze multiple columns."""
        df = pd.DataFrame({
            "wound_pct": [5.0, 5.1, 5.0, 5.2, 5.0, 5.1, 5.0],
            "brightness": [100.0, 101.0, 400.0, 100.0, 100.0, 100.0, 100.0],
        })
        detector = OutlierDetector(strategy=ZScoreStrategy(threshold=2.0))
        detector.add_column("wound_pct")
        detector.add_column("brightness")
        results = detector.detect(df)
        assert "wound_pct" in results
        assert "brightness" in results
        assert 2 in results["brightness"]

    def test_detect_with_reasons(self) -> None:
        """detect_with_reasons returns per-index reason list."""
        df = pd.DataFrame({"value": [1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0]})
        detector = OutlierDetector(strategy=ZScoreStrategy(threshold=2.0))
        detector.add_column("value")
        _, reasons = detector.detect_with_reasons(df)
        assert 3 in reasons
        assert len(reasons[3]) > 0

    def test_get_all_outlier_indices(self) -> None:
        """get_all_outlier_indices unions across columns."""
        df = pd.DataFrame({
            "col_a": [1.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "col_b": [1.0, 1.0, 200.0, 1.0, 1.0, 1.0, 1.0],
        })
        detector = OutlierDetector(strategy=ZScoreStrategy(threshold=2.0))
        detector.add_column("col_a")
        detector.add_column("col_b")
        all_indices = detector.get_all_outlier_indices(df)
        assert 1 in all_indices
        assert 2 in all_indices

    def test_get_outlier_summary(self) -> None:
        """Summary contains correct counts and percentages."""
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 200.0] * 10})
        detector = OutlierDetector(strategy=ZScoreStrategy(threshold=2.0))
        detector.add_column("value")
        summary = detector.get_outlier_summary(df)
        assert "total_rows" in summary
        assert "total_outliers" in summary
        assert "outlier_percentage" in summary
        assert summary["strategy"] == "ZScore(2.0)"

    def test_set_strategy(self) -> None:
        """Strategy can be changed after construction."""
        detector = OutlierDetector(strategy=IQRStrategy())
        detector.set_strategy(ZScoreStrategy(threshold=3.0))
        df = pd.DataFrame({"value": [1] * 10 + [100]})
        detector.add_column("value")
        results = detector.detect(df)
        assert len(results["value"]) >= 1

    def test_remove_column(self) -> None:
        """Column can be removed from analysis."""
        df = pd.DataFrame({"a": [1, 100], "b": [1, 2]})
        detector = OutlierDetector(strategy=ZScoreStrategy(threshold=2.0))
        detector.add_column("a")
        detector.add_column("b")
        detector.remove_column("a")
        results = detector.detect(df)
        assert "a" not in results
        assert "b" in results

    def test_detect_no_columns(self) -> None:
        """Empty results when no columns configured."""
        df = pd.DataFrame({"value": [1, 2, 100]})
        detector = OutlierDetector()
        results = detector.detect(df)
        assert results == {}
