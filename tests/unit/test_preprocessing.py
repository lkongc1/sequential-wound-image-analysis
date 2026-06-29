"""Unit tests for src.data.transforms.preprocessing."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms.preprocessing import (
    calculate_image_stats,
    correct_illumination,
    normalize_image,
    resize_with_padding,
)


class TestNormalizeImage:
    """Tests for normalize_image function."""

    def test_normalize_image_basic(self) -> None:
        """Normalized image stays within [alpha, beta] range."""
        img = np.array([[100, 150], [200, 250]], dtype=np.uint8)
        result = normalize_image(img, alpha=0.0, beta=255.0)

        assert result.min() >= 0.0
        assert result.max() <= 255.0

    def test_normalize_image_empty_raises(self) -> None:
        """Empty image raises ValueError."""
        with pytest.raises(ValueError, match="Empty or None image"):
            normalize_image(np.array([]))

    def test_normalize_image_preserves_shape(self) -> None:
        """Output has same shape as input."""
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        result = normalize_image(img)

        assert result.shape == img.shape


class TestCorrectIllumination:
    """Tests for correct_illumination function."""

    def test_correct_illumination_gray_method(self) -> None:
        """Gray method returns image with expected shape."""
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        result = correct_illumination(img, method="gray")

        assert result.shape == img.shape
        assert result.dtype == np.float32

    def test_correct_illumination_retinex_method(self) -> None:
        """Retinex method returns image with expected shape."""
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        result = correct_illumination(img, method="retinex")

        assert result.shape == img.shape
        assert result.dtype == np.float32

    def test_correct_illumination_unknown_method(self) -> None:
        """Unknown method returns original image."""
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        result = correct_illumination(img, method="unknown")

        # Should return original when method not recognized
        assert result.shape == img.shape


class TestResizeWithPadding:
    """Tests for resize_with_padding function."""

    def test_resize_with_padding_square_target(self) -> None:
        """Image is resized to target size with padding."""
        img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        result = resize_with_padding(img, target_size=(384, 384))

        assert result.shape == (384, 384, 3)

    def test_resize_with_padding_preserves_content(self) -> None:
        """Aspect ratio is preserved, content is centered."""
        # Use uniform white image to avoid JPEG compression artifacts
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        result = resize_with_padding(img, target_size=(384, 384))

        # Check that the center region is white (255), not black (0)
        # The resized content should be centered with padding around it
        center_region = result[92:292, 92:292]
        assert np.mean(center_region) > 200  # Should be near 255

    def test_resize_with_padding_already_correct_size(self) -> None:
        """Image already at target size is returned as-is."""
        img = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)
        result = resize_with_padding(img, target_size=(384, 384))

        assert result.shape == (384, 384, 3)


class TestCalculateImageStats:
    """Tests for calculate_image_stats function."""

    def test_calculate_image_stats_color_image(self, tmp_path: Path) -> None:
        """Color image stats are calculated correctly."""
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), img)

        stats = calculate_image_stats(img_path)

        assert stats["width"] == 256
        assert stats["height"] == 256
        assert stats["mode"] == "RGB"
        assert stats["channels"] == 3
        assert stats["is_grayscale"] is False
        assert stats["brightness_mean"] == pytest.approx(128.0, abs=1.0)
        assert stats["file_size_kb"] > 0

    def test_calculate_image_stats_grayscale(self, tmp_path: Path) -> None:
        """Grayscale image is detected correctly."""
        img = np.full((256, 256), 100, dtype=np.uint8)
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), img)

        stats = calculate_image_stats(img_path)

        assert stats["is_grayscale"] is True
        assert stats["channels"] == 1

    def test_calculate_image_stats_missing_file(self, tmp_path: Path) -> None:
        """Missing file returns default stats."""
        stats = calculate_image_stats(tmp_path / "missing.jpg")

        assert stats["width"] == 0
        assert stats["height"] == 0
        assert stats["file_size_kb"] == 0.0
