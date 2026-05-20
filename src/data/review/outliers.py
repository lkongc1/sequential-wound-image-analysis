"""Outlier detection strategies following SOLID principles.

SRP: Each outlier strategy is a single class.
OCP: Easy to add new strategies (ZScore, DBSCAN, IsolationForest).
DIP: OutlierDetector depends on BaseOutlierStrategy abstraction.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, TypeVar

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseOutlierStrategy(ABC):
    """Abstract base class for outlier detection strategies."""

    @abstractmethod
    def detect(self, df: pd.DataFrame, column: str) -> Set[int]:
        """Detect outliers in a column.

        Args:
            df: DataFrame containing the data.
            column: Column name to analyze.

        Returns:
            Set of row indices that are outliers.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return the strategy name."""
        ...


@dataclass
class IQRStrategy(BaseOutlierStrategy):
    """Interquartile Range (IQR) outlier detection.

    Outliers are values below Q1 - multiplier*IQR or above Q3 + multiplier*IQR.

    Attributes:
        multiplier: IQR multiplier (default 1.5 for standard outliers).
        include_bounds: Whether to include boundary values as outliers.
    """

    multiplier: float = 1.5
    include_bounds: bool = False

    def detect(self, df: pd.DataFrame, column: str) -> Set[int]:
        """Detect outliers using IQR method."""
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame")
            return set()

        data = df[column].dropna()
        if len(data) == 0:
            return set()

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - self.multiplier * iqr
        upper_bound = q3 + self.multiplier * iqr

        if self.include_bounds:
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
        else:
            # Only extreme outliers
            extreme_multiplier = 3.0
            extreme_lower = q1 - extreme_multiplier * iqr
            extreme_upper = q3 + extreme_multiplier * iqr
            outliers = df[(df[column] < extreme_lower) | (df[column] > extreme_upper)].index

        logger.info(
            f"IQR strategy ({self.multiplier}x): found {len(outliers)} outliers in '{column}'"
        )
        return set(outliers)

    def get_name(self) -> str:
        return f"IQR({self.multiplier})"


@dataclass
class ZScoreStrategy(BaseOutlierStrategy):
    """Z-Score outlier detection.

    Outliers are values with |z-score| > threshold.

    Attributes:
        threshold: Z-score threshold (default 3.0 for standard outliers).
    """

    threshold: float = 3.0

    def detect(self, df: pd.DataFrame, column: str) -> Set[int]:
        """Detect outliers using Z-Score method."""
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame")
            return set()

        data = df[column].dropna()
        if len(data) < 3:
            logger.warning(f"Not enough data for Z-Score in '{column}'")
            return set()

        mean = data.mean()
        std = data.std()

        if std == 0:
            logger.warning(f"Standard deviation is 0 for '{column}', cannot compute Z-Score")
            return set()

        z_scores = np.abs((data - mean) / std)
        outlier_indices = df[z_scores > self.threshold].index

        logger.info(
            f"Z-Score strategy ({self.threshold}σ): found {len(outlier_indices)} outliers in '{column}'"
        )
        return set(outlier_indices)

    def get_name(self) -> str:
        return f"ZScore({self.threshold})"


@dataclass
class PercentileStrategy(BaseOutlierStrategy):
    """Percentile-based outlier detection.

    Outliers are values outside the [lower, upper] percentile range.

    Attributes:
        lower: Lower percentile threshold (default 1).
        upper: Upper percentile threshold (default 99).
    """

    lower: float = 1.0
    upper: float = 99.0

    def detect(self, df: pd.DataFrame, column: str) -> Set[int]:
        """Detect outliers using percentile method."""
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame")
            return set()

        data = df[column].dropna()
        if len(data) == 0:
            return set()

        lower_bound = np.percentile(data, self.lower)
        upper_bound = np.percentile(data, self.upper)

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index

        logger.info(
            f"Percentile strategy [{self.lower}-{self.upper}]: "
            f"found {len(outliers)} outliers in '{column}'"
        )
        return set(outliers)

    def get_name(self) -> str:
        return f"Percentile({self.lower}-{self.upper})"


class EmptyMaskStrategy(BaseOutlierStrategy):
    """Detect outliers where mask is empty (no wound).

    This marks samples as outliers if they have an empty mask,
    which may indicate data quality issues.
    """

    def detect(self, df: pd.DataFrame, column: str = "is_empty") -> Set[int]:
        """Detect samples with empty masks."""
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame")
            return set()

        outliers = df[df[column]].index
        logger.info(f"Empty mask strategy: found {len(outliers)} outliers")
        return set(outliers)

    def get_name(self) -> str:
        return "EmptyMask"


class OutlierDetector:
    """Detects outliers in dataset using configurable strategies.

    DIP: Depends on BaseOutlierStrategy abstraction, not concrete implementations.
    OCP: Easy to add new strategies without modifying this class.

    Attributes:
        strategy: Outlier detection strategy to use.
        columns: List of columns to analyze for outliers.

    Example:
        >>> detector = OutlierDetector(strategy=IQRStrategy(multiplier=1.5))
        >>> detector.add_column("mask_wound_percentage")
        >>> detector.add_column("image_brightness_mean")
        >>> outliers = detector.detect(df)
    """

    def __init__(self, strategy: Optional[BaseOutlierStrategy] = None):
        """Initialize the outlier detector.

        Args:
            strategy: Detection strategy to use. Defaults to IQRStrategy.
        """
        self.strategy = strategy or IQRStrategy()
        self._columns: List[str] = []

    def add_column(self, column: str) -> None:
        """Add a column to analyze for outliers.

        Args:
            column: Column name to add.
        """
        if column not in self._columns:
            self._columns.append(column)

    def remove_column(self, column: str) -> None:
        """Remove a column from analysis.

        Args:
            column: Column name to remove.
        """
        if column in self._columns:
            self._columns.remove(column)

    def set_strategy(self, strategy: BaseOutlierStrategy) -> None:
        """Change the detection strategy.

        Args:
            strategy: New strategy to use.
        """
        self.strategy = strategy
        logger.info(f"Strategy changed to: {strategy.get_name()}")

    def detect(self, df: pd.DataFrame) -> Dict[str, Set[int]]:
        """Detect outliers across all configured columns.

        Args:
            df: DataFrame to analyze.

        Returns:
            Dictionary mapping column name to set of outlier row indices.
        """
        if self._columns is None or len(self._columns) == 0:
            logger.warning("No columns configured for outlier detection")
            return {}

        results: Dict[str, Set[int]] = {}
        for column in self._columns:
            results[column] = self.strategy.detect(df, column)

        return results

    def detect_with_reasons(
        self, df: pd.DataFrame
    ) -> tuple[Dict[str, Set[int]], Dict[int, List[str]]]:
        """Detect outliers and return reasons for each.

        Args:
            df: DataFrame to analyze.

        Returns:
            Tuple of (results dict, reasons dict mapping idx to list of reasons).
        """
        results = self.detect(df)
        reasons: Dict[int, List[str]] = {i: [] for i in df.index}

        for column, indices in results.items():
            for idx in indices:
                reasons[idx].append(f"{column}_{self.strategy.get_name()}")

        return results, reasons

    def get_all_outlier_indices(self, df: pd.DataFrame) -> Set[int]:
        """Get all outlier indices across all columns.

        Args:
            df: DataFrame to analyze.

        Returns:
            Set of all outlier row indices.
        """
        results = self.detect(df)
        all_indices: Set[int] = set()
        for column_indices in results.values():
            all_indices.update(column_indices)
        return all_indices

    def get_outlier_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get a summary of outlier detection results.

        Args:
            df: DataFrame to analyze.

        Returns:
            Dictionary with summary statistics.
        """
        results = self.detect(df)
        total_rows = len(df)
        all_outliers = self.get_all_outlier_indices(df)

        summary: Dict[str, Any] = {
            "strategy": self.strategy.get_name(),
            "total_rows": total_rows,
            "total_outliers": len(all_outliers),
            "outlier_percentage": (len(all_outliers) / total_rows * 100) if total_rows > 0 else 0,
            "by_column": {},
        }

        for column, indices in results.items():
            summary["by_column"][column] = {
                "count": len(indices),
                "percentage": (len(indices) / total_rows * 100) if total_rows > 0 else 0,
                "indices": sorted(list(indices))[:100],  # Limit to first 100
            }

        return summary
