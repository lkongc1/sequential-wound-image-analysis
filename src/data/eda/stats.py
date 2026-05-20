"""Statistics calculators following SOLID principles (SRP).

SRP: Each calculator has a single responsibility - only calculates statistics.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageStatisticsCalculator:
    """Calculates statistics for image files.

    Single Responsibility: Only calculates image statistics, nothing else.

    Example:
        >>> calc = ImageStatisticsCalculator()
        >>> stats = calc.calculate(Path("image.png"))
        >>> print(stats['brightness_mean'])
        127.5
    """

    def calculate(self, image_path: Path) -> Dict[str, Any]:
        """Calculate image statistics with type safety.

        Args:
            image_path: Path to the image file.

        Returns:
            Dictionary with width, height, mode, channels, brightness, contrast.
        """
        stats = {
            "width": 0, "height": 0, "mode": "UNKNOWN", "channels": 0,
            "brightness_mean": 0.0, "brightness_std": 0.0, "contrast_rms": 0.0,
            "is_grayscale": False, "file_size_kb": 0.0,
        }
        try:
            if not image_path.exists():
                return stats
            stats["file_size_kb"] = round(image_path.stat().st_size / 1024, 2)
            img_np = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img_np is None:
                return stats

            stats["height"], stats["width"] = img_np.shape[:2]
            stats["channels"] = 1 if len(img_np.shape) == 2 else img_np.shape[2]
            stats["is_grayscale"] = len(img_np.shape) == 2
            stats["mode"] = "L" if stats["is_grayscale"] else "RGB"

            if stats["is_grayscale"]:
                brightness = img_np.astype(np.float32)
                gray = img_np.astype(np.float32)
            else:
                # cv2.imread returns BGR, convert to RGB for correct luma
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                brightness = (
                    0.299 * img_rgb[:, :, 0] +
                    0.587 * img_rgb[:, :, 1] +
                    0.114 * img_rgb[:, :, 2]
                ).astype(np.float32)
                gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY).astype(np.float32)

            stats["brightness_mean"] = round(float(np.mean(brightness)), 2)
            stats["brightness_std"] = round(float(np.std(brightness)), 2)

            # Contrast RMS: RMS of gradients (Michelson-like)
            gy, gx = np.gradient(gray)
            contrast_rms = np.sqrt(gx**2 + gy**2).mean()
            stats["contrast_rms"] = round(float(contrast_rms), 2)

        except Exception as e:
            logger.warning(f"Error processing image {image_path.name}: {e}")
        return stats

    def __call__(self, image_path: Path) -> Dict[str, Any]:
        """Allow calling instance directly."""
        return self.calculate(image_path)


class MaskStatisticsCalculator:
    """Calculates statistics for mask files.

    Single Responsibility: Only calculates mask statistics, nothing else.

    Example:
        >>> calc = MaskStatisticsCalculator()
        >>> stats = calc.calculate(Path("mask.png"))
        >>> print(stats['wound_percentage'])
        25.5
    """

    def calculate(self, mask_path: Path) -> Dict[str, Any]:
        """Calculate mask statistics with type safety.

        Args:
            mask_path: Path to the mask file.

        Returns:
            Dictionary with wound_pixels, total_pixels, wound_percentage, etc.
        """
        stats = {
            "wound_pixels": 0, "total_pixels": 0, "wound_percentage": 0.0,
            "has_wound": False, "bbox_area": 0, "wound_compactness": 0.0,
            "wound_centroid_x": 0.5, "wound_centroid_y": 0.5,
            "edge_density": 0.0, "is_empty": True,
        }
        try:
            if not mask_path.exists():
                return stats
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return stats
            mask_np = mask.astype(np.float32)
            stats["total_pixels"] = mask_np.size
            wound_mask = mask_np > 0
            stats["wound_pixels"] = int(np.count_nonzero(wound_mask))
            stats["has_wound"] = bool(stats["wound_pixels"] > 0)
            stats["is_empty"] = not stats["has_wound"]

            if stats["total_pixels"] > 0:
                stats["wound_percentage"] = round(
                    (stats["wound_pixels"] / stats["total_pixels"]) * 100, 4
                )

            if stats["has_wound"]:
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
                    h, w = mask_np.shape
                    wound_rows, wound_cols = np.where(wound_mask)
                    if len(wound_rows) > 0:
                        stats["wound_centroid_x"] = round(
                            float(np.mean(wound_cols)) / max(w, 1), 4
                        )
                        stats["wound_centroid_y"] = round(
                            float(np.mean(wound_rows)) / max(h, 1), 4
                        )

            # Edge density: Canny edges / total pixels
            edges = cv2.Canny(mask_np.astype(np.uint8), 50, 150)
            edge_pixels = np.count_nonzero(edges)
            stats["edge_density"] = round(edge_pixels / stats["total_pixels"], 6) if stats["total_pixels"] > 0 else 0.0

        except Exception as e:
            logger.warning(f"Error processing mask {mask_path.name}: {e}")
        return stats

    def __call__(self, mask_path: Path) -> Dict[str, Any]:
        """Allow calling instance directly."""
        return self.calculate(mask_path)


class DatasetStatisticsCalculator:
    """Calculates aggregated statistics for the entire dataset.

    Single Responsibility: Only calculates aggregated dataset statistics.
    """

    def calculate(self, df: Any) -> Dict[str, Any]:
        """Generate dataset statistics for reporting.

        Args:
            df: DataFrame with per-file statistics.

        Returns:
            Dictionary with wound_general and by_source statistics.
        """
        stats = {}

        wound_pct = df["mask_wound_percentage"]
        stats["wound_general"] = {
            "count": len(df),
            "with_wound": int(df["mask_has_wound"].sum()),
            "without_wound": int((~df["mask_has_wound"]).sum()),
            "mean": round(float(wound_pct.mean()), 4),
            "median": round(float(wound_pct.median()), 4),
            "std": round(float(wound_pct.std()), 4),
            "min": round(float(wound_pct.min()), 4),
            "max": round(float(wound_pct.max()), 4),
        }

        stats["by_source"] = {}
        for source in df["source"].unique():
            src_df = df[df["source"] == source]
            if len(src_df) == 0:
                continue
            stats["by_source"][source] = {
                "count": len(src_df),
                "mean_wound_pct": round(float(src_df["mask_wound_percentage"].mean()), 4),
                "median_wound_pct": round(float(src_df["mask_wound_percentage"].median()), 4),
                "mean_brightness": round(float(src_df["image_brightness_mean"].mean()), 2),
            }

        return stats

    def __call__(self, df: Any) -> Dict[str, Any]:
        """Allow calling instance directly."""
        return self.calculate(df)
