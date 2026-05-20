"""File validation implementation.

Refactored from data_quality/validators.py to follow SOLID principles:
- SRP: Only handles file validation
- OCP: Implements FileValidator interface
- ISP: Specialized interface for file operations
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from src.core.base import FileValidator

logger = logging.getLogger(__name__)


class ImageMaskFileValidator(FileValidator):
    """Validates image and mask file pairs.

    Single Responsibility: Validates file existence and dimension matching.
    """

    @property
    def name(self) -> str:
        return "ImageMaskFileValidator"

    def validate(self, data: Dict[str, Path], **kwargs) -> Dict[str, Any]:
        """Validate image-mask pair.

        Args:
            data: Dictionary with 'image_path' and 'mask_path' keys

        Returns:
            Validation results
        """
        image_path = data.get("image_path")
        mask_path = data.get("mask_path")

        if not image_path or not mask_path:
            return {
                "valid": False,
                "errors": ["Missing image_path or mask_path"],
                "warnings": []
            }

        is_valid = self.validate_file_pair(image_path, mask_path)

        return {
            "valid": is_valid,
            "errors": [] if is_valid else ["File pair validation failed"],
            "warnings": []
        }

    def validate_file_pair(self, image_path: Path, mask_path: Path) -> bool:
        """Validate that both files exist and have matching dimensions.

        Args:
            image_path: Path to image file
            mask_path: Path to mask file

        Returns:
            True if both files exist and have identical dimensions
        """
        if not image_path.exists():
            logger.debug(f"Image not found: {image_path}")
            return False

        if not mask_path.exists():
            logger.debug(f"Mask not found: {mask_path}")
            return False

        try:
            img = Image.open(image_path)
            mask = Image.open(mask_path)

            if img.size != mask.size:
                logger.warning(
                    f"Dimension mismatch: image={img.size}, mask={mask.size}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating files: {e}")
            return False


class DatasetFileValidator(FileValidator):
    """Validates entire dataset file structure."""

    @property
    def name(self) -> str:
        return "DatasetFileValidator"

    def validate(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Validate dataset structure.

        Args:
            data: Dictionary with 'dataset_dir' and optional 'expected_dirs'

        Returns:
            Validation results
        """
        dataset_dir = data.get("dataset_dir")
        expected_dirs = data.get("expected_dirs", ["images", "masks"])

        if not dataset_dir:
            return {
                "valid": False,
                "errors": ["Missing dataset_dir"],
                "warnings": []
            }

        return self._validate_structure(Path(dataset_dir), expected_dirs)

    def validate_file_pair(self, image_path: Path, mask_path: Path) -> bool:
        """Delegate to ImageMaskFileValidator."""
        return ImageMaskFileValidator().validate_file_pair(image_path, mask_path)

    def _validate_structure(
        self,
        dataset_dir: Path,
        expected_dirs: list,
    ) -> Dict[str, Any]:
        """Validate dataset directory structure."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }

        if not dataset_dir.exists():
            result["valid"] = False
            result["errors"].append(f"Dataset directory not found: {dataset_dir}")
            return result

        for dir_name in expected_dirs:
            dir_path = dataset_dir / dir_name
            if not dir_path.exists():
                result["warnings"].append(f"Expected directory not found: {dir_name}")

        image_count = len(list(dataset_dir.glob("**/*.png")))
        image_count += len(list(dataset_dir.glob("**/*.jpg")))

        result["stats"]["total_images"] = image_count

        if image_count == 0:
            result["valid"] = False
            result["errors"].append("No images found in dataset")

        return result
