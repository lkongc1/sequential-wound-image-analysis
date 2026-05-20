"""Base classes for data sources.

SRP: Each source implementation has single responsibility.
OCP: New sources can be added without modifying existing code.
DIP: High-level modules depend on this abstraction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple


class BaseDataSource(ABC):
    """Abstract interface for data sources.

    Example:
        >>> source = KaggleSource(dataset_slug="user/dataset")
        >>> source.download(Path("data/raw"))
        >>> images, masks = source.load()
    """

    @abstractmethod
    def download(self, output_dir: Path, **kwargs) -> bool:
        """Download data from source."""
        pass

    @abstractmethod
    def load(self, data_dir: Path) -> Tuple[List[Path], List[Path]]:
        """Load image and mask paths from downloaded data."""
        pass

    @abstractmethod
    def validate(self, data_dir: Path) -> Dict[str, Any]:
        """Validate downloaded data integrity."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return source name for logging."""
        pass
