"""Custom exceptions for the wound segmentation project.

Following SOLID principles, each exception represents a specific error type.
"""
from typing import Any, Dict, Optional


class WoundSegError(Exception):
    """Base exception for all wound segmentation errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# =============================================================================
# DATA ERRORS
# =============================================================================

class DataError(WoundSegError):
    """Base exception for data-related errors."""
    pass


class DownloadError(DataError):
    """Raised when data download fails."""
    pass


class ExtractionError(DataError):
    """Raised when archive extraction fails."""
    pass


class ValidationError(DataError):
    """Raised when data validation fails."""
    pass


class QualityCheckError(DataError):
    """Raised when quality check fails."""
    pass


# =============================================================================
# MODEL ERRORS
# =============================================================================

class ModelError(WoundSegError):
    """Base exception for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    pass


class ModelInferenceError(ModelError):
    """Raised when model inference fails."""
    pass


# =============================================================================
# TRAINING ERRORS
# =============================================================================

class TrainingError(WoundSegError):
    """Base exception for training-related errors."""
    pass


class ConfigurationError(TrainingError):
    """Raised when configuration is invalid."""
    pass


class DeviceError(TrainingError):
    """Raised when device (GPU/CPU) issues occur."""
    pass


# =============================================================================
# API ERRORS
# =============================================================================

class APIError(WoundSegError):
    """Base exception for API-related errors."""
    pass


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    pass
