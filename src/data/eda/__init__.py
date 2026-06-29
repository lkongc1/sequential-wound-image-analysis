"""EDA module following SOLID principles.

SRP: Each sub-module has a single responsibility.
OCP: New calculators, visualizers, reporters can be added without modifying existing code.
DIP: High-level modules depend on abstractions.

Example:
    >>> from src.data.eda import ImageStatisticsCalculator, MaskStatisticsCalculator
    >>> from src.data.eda import WoundDistributionVisualizer, CSVReporter
    >>> from src.data.eda import DatasetAuditor, AuditConfig
"""
from src.data.eda.stats import (
    ImageStatisticsCalculator,
    MaskStatisticsCalculator,
    DatasetStatisticsCalculator,
)
from src.data.eda.visualizers import WoundDistributionVisualizer
from src.data.eda.reporters import CSVReporter, JSONReporter, MultiReporter
from src.data.audit.auditor import DatasetAuditor, AuditConfig

__all__ = [
    # Statistics calculators
    "ImageStatisticsCalculator",
    "MaskStatisticsCalculator",
    "DatasetStatisticsCalculator",
    # Visualizers
    "WoundDistributionVisualizer",
    # Reporters
    "CSVReporter",
    "JSONReporter",
    "MultiReporter",
    # Audit
    "DatasetAuditor",
    "AuditConfig",
]
