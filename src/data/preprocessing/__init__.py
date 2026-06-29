"""Preprocessing module for image normalization.

SOLID principles:
- SRP: Each normalizer handles one technique
- OCP: Easy to extend with new normalizers
- DIP: High-level code uses BaseNormalizer abstraction
"""
from __future__ import annotations

from src.data.preprocessing.normalizer import (
    BaseNormalizer,
    CLAHEProcessor,
    BrightnessNormalizer,
    HistogramEqualizationProcessor,
    StainNormalizationProcessor,
    create_default_normalizer,
)
from src.data.preprocessing.splitter import DataSplitter

__all__ = [
    "BaseNormalizer",
    "CLAHEProcessor",
    "BrightnessNormalizer",
    "HistogramEqualizationProcessor",
    "StainNormalizationProcessor",
    "create_default_normalizer",
    "DataSplitter",
]
