"""Mask validation implementation.

Refactored from data_quality/validators.py to follow SOLID principles:
- SRP: Only handles mask validation
- OCP: Implements MaskValidator interface
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

from src.core.base import MaskValidator

logger = logging.getLogger(__name__)


class WoundMaskValidator(MaskValidator):
    """Validates wound mask properties.

    Single Responsibility: Validates mask content, not file existence.

    Example:
        >>> validator = WoundMaskValidator()
        >>> mask = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]])
        >>> stats = validator.validate_mask(mask)
        >>> print(stats['wound_percentage'])
        33.33
    """

    def __init__(
        self,
        min_wound_ratio: float = 0.001,
        max_wound_ratio: float = 0.95,
    ):
        """Initialize mask validator.

        Args:
            min_wound_ratio: Minimum wound area ratio (default: 0.1%)
            max_wound_ratio: Maximum wound area ratio (default: 95%)
        """
        self.min_wound_ratio = min_wound_ratio
        self.max_wound_ratio = max_wound_ratio

    @property
    def name(self) -> str:
        return "WoundMaskValidator"

    def validate(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Validate mask data.

        Args:
            data: Mask array or dictionary with 'mask' key

        Returns:
            Validation results with mask statistics
        """
        if isinstance(data, dict):
            mask = data.get("mask")
        else:
            mask = data

        if mask is None:
            return {
                "valid": False,
                "errors": ["No mask provided"],
                "warnings": [],
                "stats": {}
            }

        stats = self.validate_mask(mask)

        errors = []
        if stats["wound_percentage"] < self.min_wound_ratio * 100:
            errors.append(f"Wound area too small: {stats['wound_percentage']:.2f}%")
        if stats["wound_percentage"] > self.max_wound_ratio * 100:
            errors.append(f"Wound area too large: {stats['wound_percentage']:.2f}%")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": [],
            "stats": stats
        }

    def validate_mask(self, mask: np.ndarray) -> Dict[str, Any]:
        """Validate mask properties and calculate statistics.

        Args:
            mask: Binary mask array where wound pixels > 0

        Returns:
            Dictionary with mask statistics:
            - wound_pixels: Number of wound pixels
            - total_pixels: Total pixels in mask
            - wound_percentage: Percentage of wound area
            - has_wound: Whether wound is present
            - bbox_area: Bounding box area
            - wound_compactness: Ratio of wound to bbox
            - wound_centroid_x/y: Normalized centroid coordinates
        """
        stats = {
            "wound_pixels": 0,
            "total_pixels": 0,
            "wound_percentage": 0.0,
            "has_wound": False,
            "bbox_area": 0,
            "wound_compactness": 0.0,
            "wound_centroid_x": 0.5,
            "wound_centroid_y": 0.5,
        }

        try:
            if isinstance(mask, Image.Image):
                mask = np.array(mask.convert("L"))
            elif not isinstance(mask, np.ndarray):
                mask = np.array(mask)

            stats["total_pixels"] = mask.size

            wound_mask = mask > 0
            stats["wound_pixels"] = int(np.count_nonzero(wound_mask))
            stats["has_wound"] = stats["wound_pixels"] > 0

            if stats["has_wound"]:
                stats["wound_percentage"] = round(
                    (stats["wound_pixels"] / stats["total_pixels"]) * 100, 4
                )

                rows = np.any(wound_mask, axis=1)
                cols = np.any(wound_mask, axis=0)

                if rows.any() and cols.any():
                    y1, y2 = np.where(rows)[0][[0, -1]]
                    x1, x2 = np.where(cols)[0][[0, -1]]
                    bbox_area = int((x2 - x1) * (y2 - y1))
                    stats["bbox_area"] = bbox_area

                    if bbox_area > 0:
                        stats["wound_compactness"] = round(
                            stats["wound_pixels"] / bbox_area, 4
                        )

                    h, w = mask.shape
                    wound_rows, wound_cols = np.where(wound_mask)
                    if len(wound_rows) > 0:
                        stats["wound_centroid_x"] = round(
                            float(np.mean(wound_cols)) / w, 4
                        )
                        stats["wound_centroid_y"] = round(
                            float(np.mean(wound_rows)) / h, 4
                        )

        except Exception as e:
            logger.warning(f"Error validating mask: {e}")

        return stats

    def validate_from_path(self, mask_path: Path) -> Dict[str, Any]:
        """Validate mask from file path."""
        if not mask_path.exists():
            return {
                "valid": False,
                "errors": [f"Mask file not found: {mask_path}"],
                "warnings": [],
                "stats": {}
            }

        try:
            mask = np.array(Image.open(mask_path).convert("L"))
            return self.validate(mask)
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Failed to load mask: {e}"],
                "warnings": [],
                "stats": {}
            }
