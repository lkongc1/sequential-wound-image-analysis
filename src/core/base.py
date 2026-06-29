"""Abstract base classes following SOLID principles.

This module defines the core interfaces that all components must implement.
Following Interface Segregation Principle (ISP), each interface is focused
on a single responsibility.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# DATA SOURCES
# =============================================================================

class BaseDataSource(ABC):
    """Abstract interface for data sources (OCP: easy to add new sources).

    Following Dependency Inversion Principle (DIP), high-level modules
    depend on this abstraction, not concrete implementations.

    Example:
        >>> source = KaggleSource(dataset_slug="user/dataset")
        >>> source.download(Path("data/raw"))
        >>> images, masks = source.load()
    """

    @abstractmethod
    def download(self, output_dir: Path, **kwargs) -> bool:
        """Download data from source.

        Args:
            output_dir: Directory to save downloaded data
            **kwargs: Additional source-specific parameters

        Returns:
            True if download succeeded
        """
        pass

    @abstractmethod
    def load(self, data_dir: Path) -> Tuple[List[Path], List[Path]]:
        """Load image and mask paths from downloaded data.

        Args:
            data_dir: Directory containing downloaded data

        Returns:
            Tuple of (image_paths, mask_paths)
        """
        pass

    @abstractmethod
    def validate(self, data_dir: Path) -> Dict[str, Any]:
        """Validate downloaded data integrity.

        Args:
            data_dir: Directory to validate

        Returns:
            Dictionary with validation results
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return source name for logging."""
        pass


# =============================================================================
# DOWNLOADERS (SRP: Single Responsibility)
# =============================================================================

class BaseDownloader(ABC):
    """Abstract interface for downloaders.

    Single Responsibility: Only handles downloading, not extraction or validation.
    """

    @abstractmethod
    def download(self, source: str, output_dir: Path, **kwargs) -> Path:
        """Download resource from source.

        Args:
            source: Source identifier (URL, slug, etc.)
            output_dir: Output directory

        Returns:
            Path to downloaded file
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if downloader is available (e.g., CLI installed)."""
        pass


# =============================================================================
# EXTRACTORS (SRP)
# =============================================================================

class BaseExtractor(ABC):
    """Abstract interface for extractors.

    Single Responsibility: Only handles extraction, not download.
    """

    @abstractmethod
    def extract(self, archive_path: Path, output_dir: Path, **kwargs) -> bool:
        """Extract archive to output directory.

        Args:
            archive_path: Path to archive file
            output_dir: Extraction output directory

        Returns:
            True if extraction succeeded
        """
        pass

    @abstractmethod
    def verify(self, output_dir: Path, expected_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Verify extraction integrity.

        Args:
            output_dir: Directory to verify
            expected_files: Optional list of expected files

        Returns:
            Verification results dictionary
        """
        pass


# =============================================================================
# VALIDATORS (SRP + ISP)
# =============================================================================

class BaseValidator(ABC):
    """Abstract base for all validators.

    Interface Segregation: Each validator has a single, focused method.
    """

    @abstractmethod
    def validate(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Validate data and return results.

        Args:
            data: Data to validate
            **kwargs: Additional validation parameters

        Returns:
            Dictionary with 'valid', 'errors', 'warnings' keys
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Validator name for reporting."""
        pass


class FileValidator(BaseValidator):
    """Specialized interface for file validation."""

    @abstractmethod
    def validate_file_pair(self, image_path: Path, mask_path: Path) -> bool:
        """Validate that image and mask files are compatible.

        Args:
            image_path: Path to image file
            mask_path: Path to mask file

        Returns:
            True if file pair is valid
        """
        pass


class MaskValidator(BaseValidator):
    """Specialized interface for mask validation."""

    @abstractmethod
    def validate_mask(self, mask: np.ndarray) -> Dict[str, Any]:
        """Validate mask properties.

        Args:
            mask: Mask array to validate

        Returns:
            Dictionary with mask statistics and validation status
        """
        pass


class DataFrameValidator(BaseValidator):
    """Specialized interface for DataFrame validation."""

    @abstractmethod
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate DataFrame structure and content.

        Args:
            df: DataFrame to validate

        Returns:
            Validation results
        """
        pass


# =============================================================================
# QUALITY CHECKERS (SRP)
# =============================================================================

class BaseQualityChecker(ABC):
    """Abstract interface for quality checkers.

    Single Responsibility: Each checker validates one quality aspect.
    """

    @abstractmethod
    def check(self, image_path: Path) -> Tuple[bool, Optional[str]]:
        """Check quality of an image.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (passed, error_message)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Checker name for reporting."""
        pass


class QualityPipeline(ABC):
    """Abstract interface for quality control pipeline.

    Orchestrates multiple quality checkers.
    """

    @abstractmethod
    def add_checker(self, checker: BaseQualityChecker) -> None:
        """Add a quality checker to pipeline."""
        pass

    @abstractmethod
    def run(self, image_paths: List[Path]) -> Dict[Path, List[str]]:
        """Run all quality checks on images.

        Args:
            image_paths: List of image paths to check

        Returns:
            Dictionary mapping failed paths to list of issues
        """
        pass



