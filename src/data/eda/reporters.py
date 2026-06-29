"""Report generators following SOLID principles (SRP).

SRP: Only handles report generation, nothing else.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class BaseReporter:
    """Abstract base for reporters."""

    def generate(self, df: pd.DataFrame, output_dir: Path, **kwargs) -> Dict[str, Any]:
        """Generate report and return metadata."""
        raise NotImplementedError


class CSVReporter(BaseReporter):
    """Saves enriched metadata CSV to output directory.

    Single Responsibility: Only generates CSV reports.
    """

    def generate(self, df: pd.DataFrame, output_dir: Path, **kwargs) -> Dict[str, Any]:
        """Save enriched metadata CSV.

        Args:
            df: DataFrame with per-file statistics.
            output_dir: Directory to save CSV.

        Returns:
            Dictionary with path and record count.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "metadata_enriched.csv"
        cols = [
            "filename", "original_id", "source", "split", "file_exists",
            "image_width", "image_height", "image_mode", "image_channels",
            "image_is_grayscale", "image_brightness_mean", "image_contrast",
            "mask_has_wound", "mask_wound_pixels", "mask_wound_percentage",
            "mask_bbox_area", "mask_wound_compactness",
            "mask_wound_centroid_x", "mask_wound_centroid_y",
        ]
        available = [c for c in cols if c in df.columns]
        df[available].to_csv(csv_path, index=False, encoding="utf-8")

        logger.info(f"Saved metadata CSV: {csv_path} ({len(df):,} records)")
        return {"path": str(csv_path), "records": len(df)}


class JSONReporter(BaseReporter):
    """Saves statistics as JSON report.

    Single Responsibility: Only generates JSON reports.
    """

    def generate(self, df: pd.DataFrame, output_dir: Path, **kwargs) -> Dict[str, Any]:
        """Save aggregated statistics as JSON.

        Args:
            df: DataFrame with per-file statistics.
            output_dir: Directory to save JSON.

        Returns:
            Dictionary with path and statistics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate aggregated stats
        wound_pct = df["mask_wound_percentage"]
        stats = {
            "wound_general": {
                "count": len(df),
                "with_wound": int(df["mask_has_wound"].sum()),
                "without_wound": int((~df["mask_has_wound"]).sum()),
                "mean": round(float(wound_pct.mean()), 4),
                "median": round(float(wound_pct.median()), 4),
                "std": round(float(wound_pct.std()), 4),
                "min": round(float(wound_pct.min()), 4),
                "max": round(float(wound_pct.max()), 4),
            },
            "by_source": {},
        }

        for source in df["source"].unique():
            src_df = df[df["source"] == source]
            if len(src_df) == 0:
                continue
            stats["by_source"][source] = {
                "count": len(src_df),
                "mean_wound_pct": round(float(src_df["mask_wound_percentage"].mean()), 4),
                "median_wound_pct": round(float(src_df["mask_wound_percentage"].median()), 4),
                "mean_brightness": round(float(src_df["image_brightness_mean"].mean()), 2),
            }

        json_path = output_dir / "statistics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved statistics JSON: {json_path}")
        return {"path": str(json_path), "stats": stats}


class MultiReporter:
    """Orchestrates multiple reporters.

    Single Responsibility: Coordinates multiple reporters.
    """

    def __init__(self, reporters: List[BaseReporter]):
        """Initialize with list of reporters.

        Args:
            reporters: List of reporter instances.
        """
        self.reporters = reporters

    def generate_all(self, df: pd.DataFrame, output_dir: Path, **kwargs) -> List[Dict[str, Any]]:
        """Generate reports using all reporters.

        Args:
            df: DataFrame with per-file statistics.
            output_dir: Directory to save reports.

        Returns:
            List of results from each reporter.
        """
        results = []
        for reporter in self.reporters:
            result = reporter.generate(df, output_dir, **kwargs)
            results.append(result)
        return results
