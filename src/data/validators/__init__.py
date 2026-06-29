"""Data validators module.

This module provides validators following SOLID principles:
- SRP: Each validator has a single responsibility
- OCP: New validators can be added without modifying existing code
- LSP: All validators can be substituted for their base types
- ISP: Interfaces are specific and focused
- DIP: High-level modules depend on abstractions

Example:
    >>> from src.data.validators import (
    ...     ImageMaskFileValidator,
    ...     WoundMaskValidator,
    ...     DomainShiftValidator,
    ... )
"""
from src.core.base import (
    BaseValidator,
    FileValidator,
    MaskValidator,
    DataFrameValidator,
)

from .file_validator import (
    ImageMaskFileValidator,
    DatasetFileValidator,
)

from .mask_validator import (
    WoundMaskValidator,
)

from .domain_validator import (
    DomainShiftValidator,
    OutlierValidator,
)


__all__ = [
    # Base classes
    "BaseValidator",
    "FileValidator",
    "MaskValidator",
    "DataFrameValidator",
    # File validators
    "ImageMaskFileValidator",
    "DatasetFileValidator",
    # Mask validators
    "WoundMaskValidator",
    # DataFrame validators
    "DomainShiftValidator",
    "OutlierValidator",
]
