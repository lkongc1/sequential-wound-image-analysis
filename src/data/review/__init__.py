"""Review module for dataset review and visualization.

SOLID principles:
- SRP: Each class has a single responsibility
- OCP: Reporters and detectors are extensible
- DIP: High-level modules depend on abstractions
"""
from __future__ import annotations

from src.data.review.resolvers import ImageResolver, MaskResolver
from src.data.review.outliers import (
    OutlierDetector,
    BaseOutlierStrategy,
    IQRStrategy,
    ZScoreStrategy,
    PercentileStrategy,
    EmptyMaskStrategy,
)
from src.data.review.reporters import InteractiveHTMLReporter

__all__ = [
    # Resolvers
    "ImageResolver",
    "MaskResolver",
    # Outliers
    "OutlierDetector",
    "BaseOutlierStrategy",
    "IQRStrategy",
    "ZScoreStrategy",
    "PercentileStrategy",
    "EmptyMaskStrategy",
    # Reporters
    "InteractiveHTMLReporter",
]
