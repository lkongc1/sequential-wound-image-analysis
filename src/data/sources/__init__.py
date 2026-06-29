"""Sources module following SOLID principles."""
from src.data.sources.base import BaseDataSource
from src.data.sources.kaggle_source import KaggleSource

__all__ = [
    "BaseDataSource",
    "KaggleSource",
]
