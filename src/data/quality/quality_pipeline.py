"""Quality control pipeline implementation.

Refactored from data_quality/quality_control.py to follow SOLID principles:
- SRP: Each checker validates one quality aspect
- OCP: New checkers can be added without modifying existing code
- DIP: Pipeline depends on abstractions (BaseQualityChecker)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.base import BaseQualityChecker, QualityPipeline

logger = logging.getLogger(__name__)


# =============================================================================
# INDIVIDUAL QUALITY CHECKERS (SRP)
# =============================================================================

class ResolutionChecker(BaseQualityChecker):
    """Checks if image meets minimum resolution requirements."""

    def __init__(self, min_width: int = 256, min_height: int = 256):
        self.min_width = min_width
        self.min_height = min_height

    @property
    def name(self) -> str:
        return "resolution"

    def check(self, image_path: Path) -> Tuple[bool, Optional[str]]:
        """Check if image meets resolution requirements."""
        img = cv2.imread(str(image_path))
        if img is None:
            return False, "Could not read image"

        h, w = img.shape[:2]

        if w >= self.min_width and h >= self.min_height:
            return True, None

        return False, f"Resolution {w}x{h} below minimum {self.min_width}x{self.min_height}"


class AspectRatioChecker(BaseQualityChecker):
    """Checks if aspect ratio is reasonable."""

    def __init__(self, max_ratio: float = 5.0):
        self.max_ratio = max_ratio

    @property
    def name(self) -> str:
        return "aspect_ratio"

    def check(self, image_path: Path) -> Tuple[bool, Optional[str]]:
        """Check aspect ratio."""
        img = cv2.imread(str(image_path))
        if img is None:
            return False, "Could not read image"

        h, w = img.shape[:2]
        ratio = max(w / h, h / w)

        if ratio <= self.max_ratio:
            return True, None

        return False, f"Aspect ratio {ratio:.2f} exceeds maximum {self.max_ratio}"


class BrightnessChecker(BaseQualityChecker):
    """Checks if image brightness is within acceptable range."""

    def __init__(
        self,
        min_brightness: float = 10.0,
        max_brightness: float = 240.0,
    ):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    @property
    def name(self) -> str:
        return "brightness"

    def check(self, image_path: Path) -> Tuple[bool, Optional[str]]:
        """Check brightness level."""
        img = cv2.imread(str(image_path))
        if img is None:
            return False, "Could not read image"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        if self.min_brightness <= mean_brightness <= self.max_brightness:
            return True, None

        return False, f"Brightness {mean_brightness:.1f} outside range [{self.min_brightness}, {self.max_brightness}]"


class SharpnessChecker(BaseQualityChecker):
    """Checks image sharpness using Laplacian variance."""

    def __init__(self, min_laplacian: float = 10.0):
        self.min_laplacian = min_laplacian

    @property
    def name(self) -> str:
        return "sharpness"

    def check(self, image_path: Path) -> Tuple[bool, Optional[str]]:
        """Check image sharpness."""
        img = cv2.imread(str(image_path))
        if img is None:
            return False, "Could not read image"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if laplacian_var >= self.min_laplacian:
            return True, None

        return False, f"Sharpness {laplacian_var:.1f} below minimum {self.min_laplacian}"


# =============================================================================
# QUALITY CONTROL PIPELINE (OCP + DIP)
# =============================================================================

class ImageQualityPipeline(QualityPipeline):
    """Orchestrates multiple quality checkers.

    Open/Closed Principle: New checkers can be added without modifying
    the pipeline logic.
    """

    def __init__(self):
        self._checkers: List[BaseQualityChecker] = []

    def add_checker(self, checker: BaseQualityChecker) -> None:
        """Add a quality checker to pipeline."""
        self._checkers.append(checker)
        logger.debug(f"Added checker: {checker.name}")

    def remove_checker(self, checker_name: str) -> None:
        """Remove a checker by name."""
        self._checkers = [
            c for c in self._checkers if c.name != checker_name
        ]

    def run(self, image_paths: List[Path]) -> Dict[Path, List[str]]:
        """Run all quality checks on images.

        Args:
            image_paths: List of image paths to check

        Returns:
            Dictionary mapping failed image paths to list of issues
        """
        failures: Dict[Path, List[str]] = {}

        for path in image_paths:
            issues: List[str] = []

            if not path.exists():
                failures[path] = ["file_missing"]
                continue

            for checker in self._checkers:
                passed, error = checker.check(path)
                if not passed:
                    issues.append(f"{checker.name}: {error}")

            if issues:
                failures[path] = issues

        return failures

    def run_single(self, image_path: Path) -> Tuple[bool, List[str]]:
        """Run all checks on a single image."""
        issues: List[str] = []

        if not image_path.exists():
            return False, ["file_missing"]

        for checker in self._checkers:
            passed, error = checker.check(image_path)
            if not passed:
                issues.append(f"{checker.name}: {error}")

        return len(issues) == 0, issues

    @property
    def checker_names(self) -> List[str]:
        """Return names of all registered checkers."""
        return [c.name for c in self._checkers]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_default_quality_pipeline(
    min_width: int = 256,
    min_height: int = 256,
    min_brightness: float = 10.0,
    max_brightness: float = 240.0,
    min_sharpness: float = 10.0,
    max_aspect_ratio: float = 5.0,
) -> ImageQualityPipeline:
    """Create a quality pipeline with default checkers."""
    pipeline = ImageQualityPipeline()

    pipeline.add_checker(ResolutionChecker(min_width, min_height))
    pipeline.add_checker(AspectRatioChecker(max_aspect_ratio))
    pipeline.add_checker(BrightnessChecker(min_brightness, max_brightness))
    pipeline.add_checker(SharpnessChecker(min_sharpness))

    return pipeline


# =============================================================================
# QC REPORT GENERATOR (SRP)
# =============================================================================

class QCReportGenerator:
    """Generates quality control reports from validation results."""

    @staticmethod
    def generate_report(
        df: Any,
        qc_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate Quality Control report from EDA DataFrame."""
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            return {"error": "Invalid DataFrame"}

        if qc_columns is None:
            qc_columns = [
                "qc_resolution_ok",
                "qc_brightness_ok",
                "qc_sharpness_ok",
                "qc_aspect_ratio_ok",
                "qc_overall_pass",
            ]

        report: Dict[str, Any] = {
            "total_images": len(df),
            "qc_checks": {}
        }

        for col in qc_columns:
            if col in df.columns:
                pass_count = int(df[col].sum())
                pass_rate = round(df[col].mean() * 100, 2)

                report["qc_checks"][col] = {
                    "pass_count": pass_count,
                    "fail_count": len(df) - pass_count,
                    "pass_rate": pass_rate,
                }

        if "qc_overall_pass" in df.columns:
            excluded = int((~df["qc_overall_pass"]).sum())
            report["images_for_training"] = len(df) - excluded
            report["images_to_exclude"] = excluded
            report["exclude_rate"] = round(excluded / len(df) * 100, 2)

        recommendations: List[str] = []

        for col, stats in report.get("qc_checks", {}).items():
            if stats["pass_rate"] < 90:
                recommendations.append(
                    f"LOW: {col} has only {stats['pass_rate']}% pass rate "
                    f"({stats['fail_count']} images need review)"
                )

        if not recommendations:
            recommendations.append("All QC checks passed above 90% threshold")

        report["recommendations"] = recommendations

        return report


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

def run_quality_control(
    image_paths: List[Path],
    min_width: int = 256,
    min_height: int = 256,
) -> Dict[Path, List[str]]:
    """Run full quality control pipeline."""
    pipeline = create_default_quality_pipeline(
        min_width=min_width,
        min_height=min_height
    )
    return pipeline.run(image_paths)
