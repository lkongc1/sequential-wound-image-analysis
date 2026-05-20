"""Unit tests for src.config — ComparativeConfig dataclass.

Run with: pytest tests/unit/test_config.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ComparativeConfig  # noqa: E402


class TestComparativeConfig:
    """Tests for ComparativeConfig dataclass."""

    def test_default_values(self):
        """Default values match the spec."""
        cfg = ComparativeConfig()
        assert cfg.model_names == ("unet", "attention_unet", "nested_unet", "deeplabv3")
        assert cfg.binarize_threshold == 0.5
        assert cfg.image_size == (256, 256)
        assert cfg.batch_size == 8
        assert cfg.num_workers == 4
        assert cfg.split == "test"

    def test_post_init_resolves_relative_paths(self):
        """__post_init__ resolves relative paths to absolute."""
        cfg = ComparativeConfig()
        project_root = Path(__file__).resolve().parent.parent.parent
        assert cfg.checkpoint_base_dir.is_absolute()
        assert cfg.output_dir.is_absolute()
        assert cfg.clean_csv.is_absolute()
        assert str(project_root) in str(cfg.checkpoint_base_dir)
        assert str(project_root) in str(cfg.output_dir)
        assert str(project_root) in str(cfg.clean_csv)

    def test_absolute_paths_preserved(self):
        """Absolute paths are preserved as-is."""
        abs_csv = Path("C:/absolute/path/dataset.csv")
        cfg = ComparativeConfig(clean_csv=abs_csv)
        assert cfg.clean_csv == abs_csv

    def test_custom_model_names(self):
        """Custom model_names tuple accepted."""
        cfg = ComparativeConfig(model_names=("unet", "manet"))
        assert cfg.model_names == ("unet", "manet")
        assert len(cfg.model_names) == 2

    def test_custom_thresholds(self):
        """Custom threshold and split accepted."""
        cfg = ComparativeConfig(binarize_threshold=0.75, split="val")
        assert cfg.binarize_threshold == 0.75
        assert cfg.split == "val"
