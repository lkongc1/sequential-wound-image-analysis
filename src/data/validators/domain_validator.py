"""Domain shift and outlier detection implementation.

Refactored from data_quality/validators.py to follow SOLID principles:
- SRP: Only handles domain shift and outlier detection
- OCP: Implements DataFrameValidator interface
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.core.base import DataFrameValidator

logger = logging.getLogger(__name__)


class DomainShiftValidator(DataFrameValidator):
    """Detects domain shift between data sources.

    Single Responsibility: Analyzes distribution differences between sources.

    Example:
        >>> validator = DomainShiftValidator()
        >>> result = validator.validate_dataframe(df)
        >>> if result['has_shift']:
        ...     print("Domain shift detected!")
    """

    def __init__(
        self,
        threshold_brightness: float = 30.0,
        threshold_area: float = 5.0,
    ):
        """Initialize domain shift validator.

        Args:
            threshold_brightness: Maximum allowed brightness difference
            threshold_area: Maximum allowed area difference
        """
        self.threshold_brightness = threshold_brightness
        self.threshold_area = threshold_area

    @property
    def name(self) -> str:
        return "DomainShiftValidator"

    def validate(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Validate data for domain shift.

        Args:
            data: DataFrame to validate

        Returns:
            Validation results with shift analysis
        """
        if not isinstance(data, pd.DataFrame):
            return {
                "valid": True,
                "errors": [],
                "warnings": ["Input is not a DataFrame"],
                "has_shift": False
            }

        return self.validate_dataframe(data)

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect domain shift between data sources.

        Args:
            df: DataFrame with at least 'source' column and brightness/area metrics

        Returns:
            Dictionary with:
            - by_source: Statistics per source
            - shifts: Detected shifts between sources
            - has_shift: Whether significant shift was detected
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "by_source": {},
            "shifts": {},
            "has_shift": False
        }

        if not isinstance(df, pd.DataFrame):
            result["warnings"].append("Input is not a DataFrame")
            return result

        if "source" not in df.columns:
            result["warnings"].append("DataFrame does not have 'source' column")
            return result

        sources = df["source"].unique().tolist()

        for source in sources:
            src_df = df[df["source"] == source]

            brightness = self._get_column_mean(src_df, "brightness")
            area = self._get_column_median(src_df, "wound_percentage")

            result["by_source"][source] = {
                "count": int(len(src_df)),
                "mean_brightness": round(brightness, 2),
                "median_area": round(area, 2),
            }

        sources_list = list(result["by_source"].keys())

        if len(sources_list) >= 2:
            ref = sources_list[0]

            for src in sources_list[1:]:
                d_brightness = abs(
                    result["by_source"][src]["mean_brightness"] -
                    result["by_source"][ref]["mean_brightness"]
                )
                d_area = abs(
                    result["by_source"][src]["median_area"] -
                    result["by_source"][ref]["median_area"]
                )

                has_significant_shift = (
                    d_brightness > self.threshold_brightness or
                    d_area > self.threshold_area
                )

                result["shifts"][src] = {
                    "delta_brightness": round(d_brightness, 2),
                    "delta_area": round(d_area, 2),
                    "significant": has_significant_shift,
                    "reference": ref,
                }

                if has_significant_shift:
                    result["has_shift"] = True

        return result

    def _get_column_mean(
        self,
        df: pd.DataFrame,
        base_name: str,
    ) -> float:
        """Safely get column mean with fallback column names."""
        candidates = [
            f"image_{base_name}_mean",
            f"{base_name}_mean",
            base_name,
        ]

        for col in candidates:
            if col in df.columns:
                try:
                    return float(df[col].mean())
                except (TypeError, ValueError):
                    continue

        return 0.0

    def _get_column_median(
        self,
        df: pd.DataFrame,
        base_name: str,
    ) -> float:
        """Safely get column median with fallback column names."""
        candidates = [
            f"mask_{base_name}",
            f"{base_name}",
        ]

        for col in candidates:
            if col in df.columns:
                try:
                    return float(df[col].median())
                except (TypeError, ValueError):
                    continue

        return 0.0


class OutlierValidator(DataFrameValidator):
    """Detects outliers in dataset features.

    Single Responsibility: Outlier detection only.
    """

    @property
    def name(self) -> str:
        return "OutlierValidator"

    def validate(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Validate data for outliers."""
        if not isinstance(data, pd.DataFrame):
            return {
                "valid": True,
                "errors": [],
                "warnings": ["Input is not a DataFrame"],
            }

        columns = kwargs.get("columns")
        return self.validate_dataframe(data, columns)

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
    ) -> Dict[str, Any]:
        """Get summary of outliers using IQR method.

        Args:
            df: DataFrame to analyze
            columns: Specific columns to check (default: all numeric)

        Returns:
            Dictionary with outlier statistics
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "total_rows": len(df),
            "outliers_per_column": {},
        }

        if not isinstance(df, pd.DataFrame):
            return result

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                continue

            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                outliers = (df[col] < lower) | (df[col] > upper)
                count = int(outliers.sum())
                rate = round(count / len(df) * 100, 2) if len(df) > 0 else 0.0

                result["outliers_per_column"][col] = {
                    "count": count,
                    "rate": rate,
                    "lower_bound": round(lower, 4),
                    "upper_bound": round(upper, 4),
                    "Q1": round(Q1, 4),
                    "Q3": round(Q3, 4),
                    "IQR": round(IQR, 4),
                }
            except Exception as e:
                logger.debug(f"Could not analyze column {col}: {e}")

        return result
