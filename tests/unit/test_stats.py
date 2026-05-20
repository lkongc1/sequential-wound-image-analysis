"""Unit tests for src.data.eda.stats."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eda.stats import (
    DatasetStatisticsCalculator,
    ImageStatisticsCalculator,
    MaskStatisticsCalculator,
)


class TestImageStatisticsCalculator:
    """Tests for ImageStatisticsCalculator."""

    def test_calculate_grayscale_image(self, tmp_path: Path) -> None:
        """Grayscale image returns correct brightness stats."""
        img = np.full((100, 100), 128, dtype=np.uint8)
        img_path = tmp_path / "gray.png"
        cv2.imwrite(str(img_path), img)

        calc = ImageStatisticsCalculator()
        stats = calc.calculate(img_path)

        assert stats["brightness_mean"] == 128.0
        assert stats["brightness_std"] == 0.0
        assert stats["is_grayscale"] is True
        assert stats["width"] == 100
        assert stats["height"] == 100

    def test_calculate_color_image(self, tmp_path: Path) -> None:
        """Color image returns brightness computed from luma."""
        # cv2.imwrite uses BGR; [200,150,100] BGR corresponds to [100,150,200] RGB
        img = np.full((100, 100, 3), [200, 150, 100], dtype=np.uint8)
        img_path = tmp_path / "color.png"
        cv2.imwrite(str(img_path), img)

        calc = ImageStatisticsCalculator()
        stats = calc.calculate(img_path)

        # Y = 0.299*R + 0.587*G + 0.114*B
        expected = 0.299 * 100 + 0.587 * 150 + 0.114 * 200
        assert abs(stats["brightness_mean"] - expected) < 1.0
        assert stats["is_grayscale"] is False
        assert stats["channels"] == 3

    def test_calculate_nonexistent_file(self, tmp_path: Path) -> None:
        """Nonexistent file returns default stats."""
        calc = ImageStatisticsCalculator()
        stats = calc.calculate(tmp_path / "missing.png")

        assert stats["brightness_mean"] == 0.0
        assert stats["brightness_std"] == 0.0
        assert stats["width"] == 0

    def test_calculate_varying_brightness(self, tmp_path: Path) -> None:
        """Image with varying brightness has non-zero std."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, :50] = 100
        img[:, 50:] = 200
        img_path = tmp_path / "varying.png"
        cv2.imwrite(str(img_path), img)

        calc = ImageStatisticsCalculator()
        stats = calc.calculate(img_path)

        assert stats["brightness_std"] > 0
        assert stats["brightness_mean"] > 0

    def test_calculate_contrast_rms(self, tmp_path: Path) -> None:
        """Contrast RMS is computed from gradient magnitude."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, :50] = 0
        img[:, 50:] = 255
        img_path = tmp_path / "contrast.png"
        cv2.imwrite(str(img_path), img)

        calc = ImageStatisticsCalculator()
        stats = calc.calculate(img_path)

        assert stats["contrast_rms"] > 0

    def test_callable_interface(self, tmp_path: Path) -> None:
        """Calculator is callable."""
        img = np.full((50, 50), 100, dtype=np.uint8)
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), img)

        calc = ImageStatisticsCalculator()
        stats = calc(img_path)
        assert stats["brightness_mean"] == 100.0


class TestMaskStatisticsCalculator:
    """Tests for MaskStatisticsCalculator."""

    def test_calculate_empty_mask(self, tmp_path: Path) -> None:
        """Empty mask returns zero wound pixels and is_empty=True."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask_path = tmp_path / "empty.png"
        cv2.imwrite(str(mask_path), mask)

        calc = MaskStatisticsCalculator()
        stats = calc.calculate(mask_path)

        assert stats["wound_pixels"] == 0
        assert stats["has_wound"] is False
        assert stats["is_empty"] is True
        assert stats["wound_percentage"] == 0.0

    def test_calculate_mask_with_wound(self, tmp_path: Path) -> None:
        """Mask with wound returns correct percentage."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255  # 50x50 = 2500 pixels out of 10000
        mask_path = tmp_path / "wound.png"
        cv2.imwrite(str(mask_path), mask)

        calc = MaskStatisticsCalculator()
        stats = calc.calculate(mask_path)

        assert stats["wound_pixels"] == 2500
        assert stats["has_wound"] is True
        assert stats["is_empty"] is False
        assert 24.0 < stats["wound_percentage"] < 26.0  # ~25%

    def test_calculate_nonexistent_file(self, tmp_path: Path) -> None:
        """Nonexistent file returns default stats."""
        calc = MaskStatisticsCalculator()
        stats = calc.calculate(tmp_path / "missing.png")

        assert stats["wound_pixels"] == 0
        assert stats["is_empty"] is True

    def test_calculate_wound_centroid(self, tmp_path: Path) -> None:
        """Centroid is normalized to [0, 1] range."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255  # centered at 50,50
        mask_path = tmp_path / "centroid.png"
        cv2.imwrite(str(mask_path), mask)

        calc = MaskStatisticsCalculator()
        stats = calc.calculate(mask_path)

        # Centroid should be near 0.5 (center of 100x100 image)
        assert 0.45 < stats["wound_centroid_x"] < 0.55
        assert 0.45 < stats["wound_centroid_y"] < 0.55

    def test_calculate_edge_density(self, tmp_path: Path) -> None:
        """Edge density is ratio of edge pixels to total pixels."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255  # solid square with edges
        mask_path = tmp_path / "edge.png"
        cv2.imwrite(str(mask_path), mask)

        calc = MaskStatisticsCalculator()
        stats = calc.calculate(mask_path)

        assert stats["edge_density"] > 0
        assert stats["edge_density"] < 1.0

    def test_calculate_compactness(self, tmp_path: Path) -> None:
        """Compactness is wound_area / bbox_area."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255  # solid 50x50 square
        mask_path = tmp_path / "compact.png"
        cv2.imwrite(str(mask_path), mask)

        calc = MaskStatisticsCalculator()
        stats = calc.calculate(mask_path)

        # For a solid square, compactness ~= area/bbox = 1.0
        assert stats["wound_compactness"] > 0.9

    def test_callable_interface(self, tmp_path: Path) -> None:
        """Calculator is callable."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask_path = tmp_path / "test.png"
        cv2.imwrite(str(mask_path), mask)

        calc = MaskStatisticsCalculator()
        stats = calc(mask_path)
        assert stats["is_empty"] is True


class TestDatasetStatisticsCalculator:
    """Tests for DatasetStatisticsCalculator."""

    def test_calculate_basic_stats(self) -> None:
        """Calculates correct wound percentage statistics."""
        df = pd.DataFrame({
            "source": ["medetec", "fusc", "medetec", "wsnet"],
            "mask_wound_percentage": [10.0, 20.0, 15.0, 25.0],
            "mask_has_wound": [True, True, True, True],
            "image_brightness_mean": [100.0, 120.0, 110.0, 130.0],
        })

        calc = DatasetStatisticsCalculator()
        stats = calc.calculate(df)

        assert stats["wound_general"]["count"] == 4
        assert stats["wound_general"]["with_wound"] == 4
        assert abs(stats["wound_general"]["mean"] - 17.5) < 0.1

    def test_calculate_by_source(self) -> None:
        """Groups statistics by source."""
        df = pd.DataFrame({
            "source": ["medetec", "medetec", "fusc", "fusc"],
            "mask_wound_percentage": [10.0, 12.0, 20.0, 22.0],
            "mask_has_wound": [True, True, True, True],
            "image_brightness_mean": [100.0, 102.0, 120.0, 122.0],
        })

        calc = DatasetStatisticsCalculator()
        stats = calc.calculate(df)

        assert "by_source" in stats
        assert "medetec" in stats["by_source"]
        assert "fusc" in stats["by_source"]
        assert stats["by_source"]["medetec"]["count"] == 2
        assert abs(stats["by_source"]["medetec"]["mean_wound_pct"] - 11.0) < 0.1

    def test_calculate_empty_dataframe(self) -> None:
        """Empty DataFrame returns empty results."""
        df = pd.DataFrame({
            "source": [],
            "mask_wound_percentage": [],
            "mask_has_wound": [],
            "image_brightness_mean": [],
        })

        calc = DatasetStatisticsCalculator()
        stats = calc.calculate(df)

        assert stats["wound_general"]["count"] == 0


# Import pandas for test
import pandas as pd
