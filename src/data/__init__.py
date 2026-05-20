"""Data pipeline following SOLID principles.

Facade module re-exporting all public symbols for backwards compatibility.

New structure:
- eda: Statistics calculators, visualizers, reporters
- audit: Dataset auditor with DIP
- quality: Quality checkers and pipeline
- sources: Data source implementations
- review: Dataset review (resolvers, outliers, reporters)
- preprocessing: Image normalization processors
"""
from __future__ import annotations

# =============================================================================
# SUBMODULE IMPORTS (new structure)
# =============================================================================

# EDA subpackage
from src.data.eda import (
    ImageStatisticsCalculator,
    MaskStatisticsCalculator,
    DatasetStatisticsCalculator,
    WoundDistributionVisualizer,
    CSVReporter,
    JSONReporter,
)

# Audit subpackage
from src.data.audit import DatasetAuditor, AuditConfig

# Quality subpackage
from src.data.quality import (
    ImageQualityPipeline,
    ResolutionChecker,
    AspectRatioChecker,
    BrightnessChecker,
    SharpnessChecker,
    run_quality_control,
    create_default_quality_pipeline,
)

# Sources subpackage
from src.data.sources import KaggleSource

# Review subpackage (NEW)
from src.data.review import (
    ImageResolver,
    MaskResolver,
    OutlierDetector,
    InteractiveHTMLReporter,
)

# Preprocessing subpackage (NEW)
from src.data.preprocessing import (
    CLAHEProcessor,
    BrightnessNormalizer,
    create_default_normalizer,
)

# Transforms subpackage
from src.data.transforms import (
    normalize_image,
    correct_illumination,
    resize_with_padding,
    calculate_image_stats,
    get_training_augmentation,
    get_inference_augmentation,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # EDA
    "ImageStatisticsCalculator",
    "MaskStatisticsCalculator",
    "DatasetStatisticsCalculator",
    "WoundDistributionVisualizer",
    "CSVReporter",
    "JSONReporter",
    # Audit
    "DatasetAuditor",
    "AuditConfig",
    # Quality
    "ImageQualityPipeline",
    "ResolutionChecker",
    "AspectRatioChecker",
    "BrightnessChecker",
    "SharpnessChecker",
    "run_quality_control",
    "create_default_quality_pipeline",
    # Sources
    "KaggleSource",
    # Review
    "ImageResolver",
    "MaskResolver",
    "OutlierDetector",
    "InteractiveHTMLReporter",
    # Preprocessing
    "CLAHEProcessor",
    "BrightnessNormalizer",
    "create_default_normalizer",
    # Transforms
    "normalize_image",
    "correct_illumination",
    "resize_with_padding",
    "calculate_image_stats",
    "get_training_augmentation",
    "get_inference_augmentation",
]
