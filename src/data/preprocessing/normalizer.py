"""Image normalization processors following SOLID principles.

SRP: Each processor handles a single normalization technique.
OCP: Easy to add new normalizers (Stain normalization, histogram equalization).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BaseNormalizer(ABC):
    """Abstract base class for image normalizers."""

    @abstractmethod
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize an image.

        Args:
            image: Input image as numpy array (BGR or RGB).

        Returns:
            Normalized image.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return the normalizer name."""
        ...


@dataclass
class CLAHEProcessor(BaseNormalizer):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE) processor.

    SRP: Handles only CLAHE normalization for brightness/contrast.
    OCP: Clip limit and tile size are configurable.

    Attributes:
        clip_limit: Threshold for contrast limiting (default 2.0).
        tile_size: Size of grid for histogram equalization (default 8).
        clip_histogram: Whether to clip histogram before equalization.

    Example:
        >>> processor = CLAHEProcessor(clip_limit=2.0, tile_size=8)
        >>> normalized = processor.normalize(image)
    """

    clip_limit: float = 2.0
    tile_size: int = 8
    clip_histogram: bool = True

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE normalization to the image.

        Args:
            image: Input image in BGR format (OpenCV default).

        Returns:
            Image with enhanced contrast.
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to CLAHEProcessor")
            return image

        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_size, self.tile_size)
        )
        l_enhanced = clahe.apply(l_channel)

        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        logger.debug(f"CLAHE applied: clip={self.clip_limit}, tile={self.tile_size}")
        return result

    def get_name(self) -> str:
        return f"CLAHE({self.clip_limit}, {self.tile_size})"


@dataclass
class BrightnessNormalizer(BaseNormalizer):
    """Normalize image brightness to a target mean.

    SRP: Handles only brightness normalization.
    Useful for standardizing illumination across images.

    Attributes:
        target_mean: Target mean brightness value (0-255).
        clip_values: Whether to clip values to valid range.

    Example:
        >>> normalizer = BrightnessNormalizer(target_mean=128.0)
        >>> normalized = normalizer.normalize(image)
    """

    target_mean: float = 128.0
    clip_values: bool = True

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize brightness to target mean.

        Args:
            image: Input image in BGR format.

        Returns:
            Brightness-normalized image.
        """
        if image is None or image.size == 0:
            return image

        # Convert to LAB and get L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Calculate current mean and scaling factor
        current_mean = np.mean(l_channel)
        if current_mean == 0:
            logger.warning("Current mean is 0, returning original image")
            return image

        scale_factor = self.target_mean / current_mean

        # Scale the L channel
        l_scaled = np.clip(l_channel * scale_factor, 0, 255).astype(np.uint8)

        # Merge and convert back
        normalized = cv2.merge([l_scaled, a_channel, b_channel])
        result = cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)

        logger.debug(f"Brightness normalized: {current_mean:.1f} -> {self.target_mean:.1f}")
        return result

    def get_name(self) -> str:
        return f"BrightnessNorm({self.target_mean})"


@dataclass
class HistogramEqualizationProcessor(BaseNormalizer):
    """Apply histogram equalization for overall contrast enhancement.

    Simpler than CLAHE but effective for uniformly distributed histograms.
    Works on grayscale or individual channels.
    """

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization.

        Args:
            image: Input image in BGR format.

        Returns:
            Contrast-enhanced image.
        """
        if image is None or image.size == 0:
            return image

        # Convert to YCrCb and equalize Y channel
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        y_equalized = cv2.equalizeHist(y)

        equalized = cv2.merge([y_equalized, cr, cb])
        result = cv2.cvtColor(equalized, cv2.COLOR_YCrCb2BGR)

        logger.debug("Histogram equalization applied")
        return result

    def get_name(self) -> str:
        return "HistogramEq"


@dataclass
class StainNormalizationProcessor(BaseNormalizer):
    """Stain normalization for histopathology images.

    This is a placeholder for more advanced stain normalization.
    Basic implementation normalizes color statistics.

    Attributes:
        target_stats: Target color statistics (mean, std per channel).
        reference_image: Optional reference image to extract stats from.
    """

    target_stats: Optional[dict] = None

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply basic stain normalization.

        Args:
            image: Input image in BGR format.

        Returns:
            Stain-normalized image.
        """
        if image is None or image.size == 0:
            return image

        result = image.copy()

        if self.target_stats:
            for i in range(3):  # B, G, R channels
                channel = result[:, :, i].astype(np.float32)
                mean = np.mean(channel)
                std = np.std(channel)
                target_mean = self.target_stats.get(f"mean_{i}", 128)
                target_std = self.target_stats.get(f"std_{i}", 64)

                if std > 0:
                    normalized = ((channel - mean) / std) * target_std + target_mean
                    result[:, :, i] = np.clip(normalized, 0, 255).astype(np.uint8)

        logger.debug("Stain normalization applied")
        return result

    def get_name(self) -> str:
        return "StainNorm"


def create_default_normalizer() -> CLAHEProcessor:
    """Create the default normalizer for the project.

    Returns:
        CLAHEProcessor with default parameters.
    """
    return CLAHEProcessor(clip_limit=2.0, tile_size=8)
