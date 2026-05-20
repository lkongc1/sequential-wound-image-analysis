"""Centralized configuration dataclasses for wound segmentation project.

Usage:
    from src.config import QualityConfig, PathsConfig

    quality = QualityConfig(min_resolution=(256, 256))
    paths = PathsConfig()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class ComparativeConfig:
    """Configuration for comparative model evaluation.

    Attributes:
        model_names: Registered model names to evaluate.
        checkpoint_base_dir: Base directory containing per-model checkpoint subdirs.
        output_dir: Directory for evaluation reports and heatmaps.
        binarize_threshold: Threshold for binarizing prediction probabilities.
        clean_csv: Path to cleaned dataset CSV (dataset_final.csv).
        image_size: Target (height, width) for resizing images.
        batch_size: Batch size for inference.
        num_workers: DataLoader worker processes.
        split: Dataset split to use for evaluation ("test" or "val").
    """
    model_names: tuple[str, ...] = ("unet", "attention_unet", "nested_unet", "deeplabv3")
    checkpoint_base_dir: Path = field(default_factory=lambda: Path("models/checkpoints"))
    output_dir: Path = field(default_factory=lambda: Path("reports/comparative"))
    binarize_threshold: float = 0.5
    clean_csv: Path = field(default_factory=lambda: Path("data/processed/dataset_final.csv"))
    image_size: tuple[int, int] = (256, 256)
    batch_size: int = 8
    num_workers: int = 4
    split: str = "test"

    def __post_init__(self) -> None:
        """Resolve relative paths to absolute paths."""
        project_root = Path(__file__).parent.parent
        if not self.checkpoint_base_dir.is_absolute():
            self.checkpoint_base_dir = project_root / self.checkpoint_base_dir
        if not self.output_dir.is_absolute():
            self.output_dir = project_root / self.output_dir
        if not self.clean_csv.is_absolute():
            self.clean_csv = project_root / self.clean_csv


@dataclass
class QualityConfig:
    """Image quality control thresholds.

    Attributes:
        min_resolution: Minimum (width, height) in pixels.
        max_aspect_ratio: Maximum aspect ratio (width/height or height/width).
        brightness_range: (min, max) acceptable mean brightness values.
        min_laplacian: Minimum Laplacian variance for sharpness.
        min_wound_ratio: Minimum wound area as fraction of total pixels.
    """
    min_resolution: Tuple[int, int] = (256, 256)
    max_aspect_ratio: float = 5.0
    brightness_range: Tuple[float, float] = (10.0, 240.0)
    min_laplacian: float = 10.0
    min_wound_ratio: float = 0.01


@dataclass
class PathsConfig:
    """Path configuration for wound segmentation project.

    Attributes:
        project_root: Root directory of the project.
        data_dir: Path to raw wound segmentation data.
        output_dir: Path for EDA and processing outputs.
        models_dir: Path to saved models and checkpoints.
    """
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path("data/raw/data_wound_seg"))
    output_dir: Path = field(default_factory=lambda: Path("output/eda"))
    models_dir: Path = field(default_factory=lambda: Path("models"))

    def __post_init__(self) -> None:
        """Resolve relative paths to absolute paths."""
        if not self.data_dir.is_absolute():
            self.data_dir = self.project_root / self.data_dir
        if not self.output_dir.is_absolute():
            self.output_dir = self.project_root / self.output_dir
        if not self.models_dir.is_absolute():
            self.models_dir = self.project_root / self.models_dir


@dataclass
class EDAConfig:
    """EDA (Exploratory Data Analysis) configuration.

    Attributes:
        max_workers: Number of parallel workers for file processing.
        chunk_size: Number of files to process per batch.
        outlier_iqr_multiplier: IQR multiplier for outlier detection.
    """
    max_workers: int = 8
    chunk_size: int = 100
    outlier_iqr_multiplier: float = 1.5


@dataclass
class TrainingConfig:
    """Training configuration and paths.

    Attributes:
        cleaned_csv: Path to cleaned dataset CSV.
        checkpoint_dir: Directory for model checkpoints.
        image_size: Target (height, width) for training images.
        batch_size: Training batch size.
        max_epochs: Maximum training epochs.
        num_workers: DataLoader worker processes.
        learning_rate: Optimizer learning rate.
        encoder_name: Backbone encoder for the model.
        val_split_size: Fraction of training data for validation (default 0.2 = 20%).
        accelerator: Device to use for training ("auto", "gpu", "cpu").
        devices: Number of devices to use.
        precision: Training precision ("16-mixed" for GPU, "32" for full precision).
    """
    cleaned_csv: Path = field(default_factory=lambda: Path("data/reports/dataset_final.csv"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("models/checkpoints/unet_resnet50_mvp"))
    image_size: Tuple[int, int] = (256, 256)
    batch_size: int = 8
    max_epochs: int = 50
    num_workers: int = 4
    learning_rate: float = 1e-4
    encoder_name: str = "resnet50"
    val_split_size: float = 0.2
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "16-mixed"

    def __post_init__(self) -> None:
        """Resolve relative paths to absolute paths."""
        project_root = Path(__file__).parent.parent
        if not self.cleaned_csv.is_absolute():
            self.cleaned_csv = project_root / self.cleaned_csv
        if not self.checkpoint_dir.is_absolute():
            self.checkpoint_dir = project_root / self.checkpoint_dir
