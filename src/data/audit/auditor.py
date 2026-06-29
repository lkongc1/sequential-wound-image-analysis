"""Dataset auditor following SOLID principles (DIP).

DIP: High-level module depends on abstractions, not concrete implementations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

if TYPE_CHECKING:
    from src.data.eda.stats import ImageStatisticsCalculator, MaskStatisticsCalculator
    from src.data.eda.visualizers import WoundDistributionVisualizer
    from src.data.eda.reporters import CSVReporter

logger = logging.getLogger(__name__)


@dataclass
class AuditConfig:
    """Configuration for dataset audit."""
    data_dir: Path
    output_dir: Path
    train_images_dir: Optional[Path] = None
    train_masks_dir: Optional[Path] = None
    test_images_dir: Optional[Path] = None
    test_masks_dir: Optional[Path] = None
    correspondence_file: Optional[Path] = None
    max_workers: int = 8

    def __post_init__(self):
        """Set default paths based on data_dir."""
        if self.train_images_dir is None:
            self.train_images_dir = self.data_dir / "train_images"
        if self.train_masks_dir is None:
            self.train_masks_dir = self.data_dir / "train_masks"
        if self.test_images_dir is None:
            self.test_images_dir = self.data_dir / "test_images"
        if self.test_masks_dir is None:
            self.test_masks_dir = self.data_dir / "test_masks"
        if self.correspondence_file is None:
            self.correspondence_file = self.data_dir / "correspondence_table.xlsx"


class DatasetAuditor:
    """Orchestrates the complete dataset audit pipeline.

    DIP: Depends on abstractions (calculators, visualizers, reporters)
    rather than concrete implementations.

    Example:
        >>> auditor = DatasetAuditor(
        ...     config=AuditConfig(data_dir=Path("data"), output_dir=Path("output")),
        ...     image_calculator=ImageStatisticsCalculator(),
        ...     mask_calculator=MaskStatisticsCalculator(),
        ...     visualizer=WoundDistributionVisualizer(Path("output")),
        ...     csv_reporter=CSVReporter(),
        ... )
        >>> df = auditor.run()
    """

    def __init__(
        self,
        config: AuditConfig,
        image_calculator: "ImageStatisticsCalculator",
        mask_calculator: "MaskStatisticsCalculator",
        visualizer: "WoundDistributionVisualizer",
        csv_reporter: "CSVReporter",
    ):
        """Initialize auditor with injected dependencies.

        Args:
            config: Audit configuration.
            image_calculator: Calculates image statistics.
            mask_calculator: Calculates mask statistics.
            visualizer: Generates visualizations.
            csv_reporter: Generates CSV reports.
        """
        self.config = config
        self.image_calc = image_calculator
        self.mask_calc = mask_calculator
        self.visualizer = visualizer
        self.csv_reporter = csv_reporter

        self.df: Optional[pd.DataFrame] = None
        self.df_metadata: Optional[pd.DataFrame] = None
        self.stats: Dict[str, Any] = {}

        logger.info(
            f"Auditor initialized | Data: {config.data_dir} | Output: {config.output_dir}"
        )

    def load_correspondence_table(self) -> pd.DataFrame:
        """Load and preprocess correspondence table."""
        logger.info("Loading correspondence table...")
        if not self.config.correspondence_file.exists():
            raise FileNotFoundError(f"Not found: {self.config.correspondence_file}")
        self.df = pd.read_excel(self.config.correspondence_file)
        logger.info(f"Loaded {len(self.df):,} records")

        def extract_source(filename: str) -> str:
            if pd.isna(filename) or "_" not in str(filename):
                return "unknown"
            return str(filename).split("_")[0].lower()

        self.df["source"] = self.df["new_id"].apply(extract_source)
        return self.df

    def process_all_files(self) -> pd.DataFrame:
        """Process all files in parallel with ThreadPoolExecutor."""
        if self.df is None:
            self.load_correspondence_table()

        results: List[dict] = []

        def process_row(row: pd.Series) -> Optional[dict]:
            try:
                filename = row["new_id"]
                if (self.config.train_images_dir / filename).exists():
                    split = "train"
                    img_path = self.config.train_images_dir / filename
                    mask_path = self.config.train_masks_dir / filename
                else:
                    split = "test"
                    img_path = self.config.test_images_dir / filename
                    mask_path = self.config.test_masks_dir / filename

                img_stats = self.image_calc.calculate(img_path)
                mask_stats = self.mask_calc.calculate(mask_path)

                return {
                    "filename": filename,
                    "original_id": row.get("origin_id", ""),
                    "source": row.get("source", "unknown"),
                    "split": split,
                    "file_exists": bool(img_path.exists() and mask_path.exists()),
                    **{f"image_{k}": v for k, v in img_stats.items()},
                    **{f"mask_{k}": v for k, v in mask_stats.items()},
                }
            except Exception as e:
                logger.debug(f"Error processing row {row.get('new_id', 'unknown')}: {e}")
                return None

        logger.info("Processing all files...")
        if self.df is None:
            raise ValueError("DataFrame is None")

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(process_row, row) for _, row in self.df.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                result = future.result()
                if result:
                    results.append(result)

        self.df_metadata = pd.DataFrame(results)
        logger.info(f"Processed {len(self.df_metadata):,} files")
        return self.df_metadata

    def generate_statistics(self) -> Dict[str, Any]:
        """Generate dataset statistics."""
        if self.df_metadata is None:
            self.process_all_files()
        assert self.df_metadata is not None
        # Statistics are now calculated by individual calculators
        # This method kept for backwards compatibility
        return self.stats

    def run(self, skip_plots: bool = False) -> pd.DataFrame:
        """Execute full audit pipeline.

        Args:
            skip_plots: If True, skip visualization generation.

        Returns:
            DataFrame with per-file metadata.
        """
        logger.info("Starting audit pipeline...")
        self.load_correspondence_table()
        self.process_all_files()
        self.visualizer.visualize(self.df_metadata, skip_plots=skip_plots)
        self.csv_reporter.generate(self.df_metadata, self.config.output_dir)
        logger.info("Audit pipeline completed")
        return self.df_metadata if self.df_metadata is not None else pd.DataFrame()
