"""Tests for instance segmentation (watershed + metrics).

Run with: pytest tests/unit/test_instance_seg.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import InstanceConfig  # noqa: E402
from src.inference.postprocessing import (  # noqa: E402
    compute_instance_stats,
    watershed_instances,
)
from src.metrics.segmentation import per_instance_dice  # noqa: E402


# ============================================================
# InstanceConfig
# ============================================================

class TestInstanceConfig:
    """Tests for InstanceConfig dataclass."""

    def test_default_values(self):
        """Default values match the spec."""
        cfg = InstanceConfig()
        assert cfg.dist_threshold_ratio == 0.3
        assert cfg.min_instance_area_px == 50
        assert cfg.kernel_close == 3
        assert cfg.polygon_epsilon_ratio == 0.001
        assert cfg.polygon_max_points == 32
        assert cfg.yolo_image_size == 640
        assert cfg.yolo_confidence == 0.25

    def test_custom_values(self):
        """Custom values override defaults."""
        cfg = InstanceConfig(dist_threshold_ratio=0.5, min_instance_area_px=100)
        assert cfg.dist_threshold_ratio == 0.5
        assert cfg.min_instance_area_px == 100
        assert cfg.kernel_close == 3  # unchanged


# ============================================================
# watershed_instances
# ============================================================

class TestWatershedInstances:
    """Tests for watershed_instances()."""

    def test_empty_mask(self):
        """Empty mask returns zero-filled instance map."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = watershed_instances(mask)
        assert result.shape == (100, 100)
        assert result.dtype == np.uint16
        assert result.max() == 0

    def test_single_blob(self):
        """Single blob returns a single instance."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:60] = 255
        result = watershed_instances(mask, min_instance_area=10)
        assert result.dtype == np.uint16
        # Should have exactly 1 unique positive id
        unique = np.unique(result)
        assert len(unique[unique > 0]) == 1

    def test_two_separate_blobs(self):
        """Two disconnected blobs return 2 instances."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 255   # blob 1
        mask[60:80, 60:80] = 255   # blob 2
        result = watershed_instances(mask, min_instance_area=10)
        unique = np.unique(result)
        instance_ids = unique[unique > 0]
        assert len(instance_ids) == 2

    def test_min_instance_area_filter(self):
        """Blobs smaller than min_instance_area are removed."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 255   # 100 px — kept
        mask[80:85, 80:85] = 255   # 25 px — removed if min=50
        result = watershed_instances(mask, min_instance_area=50)
        unique = np.unique(result)
        instance_ids = unique[unique > 0]
        assert len(instance_ids) == 1

    def test_touching_circles_via_watershed(self):
        """Two circles with thin bridge should be separated by watershed."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        yy, xx = np.ogrid[:100, :100]
        c1 = (xx - 25)**2 + (yy - 30)**2 <= 20**2
        c2 = (xx - 75)**2 + (yy - 30)**2 <= 20**2
        # Thin 2-pixel bridge at the narrowest point
        bridge = (xx >= 44) & (xx <= 56) & (yy >= 29) & (yy <= 31)
        mask[c1 | c2 | bridge] = 255

        result = watershed_instances(mask, dist_threshold_ratio=0.3, min_instance_area=10, kernel_close=1)
        unique = np.unique(result)
        instance_ids = unique[unique > 0]
        assert len(instance_ids) == 2, f"Expected 2 instances, got {len(instance_ids)}"

    def test_float_input_mask(self):
        """Float mask (0.0-1.0) is handled correctly."""
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[10:30, 10:30] = 1.0
        result = watershed_instances(mask, min_instance_area=10)
        assert result.dtype == np.uint16
        unique = np.unique(result)
        assert len(unique[unique > 0]) == 1


# ============================================================
# compute_instance_stats
# ============================================================

class TestComputeInstanceStats:
    """Tests for compute_instance_stats()."""

    def test_empty_map(self):
        """Empty instance map returns empty list."""
        instance_map = np.zeros((100, 100), dtype=np.uint16)
        stats = compute_instance_stats(instance_map, (100, 100))
        assert stats == []

    def test_single_instance(self):
        """Single instance returns correct stats."""
        instance_map = np.zeros((100, 100), dtype=np.uint16)
        instance_map[20:40, 30:60] = 1  # 20x30 = 600px
        stats = compute_instance_stats(instance_map, (100, 100))
        assert len(stats) == 1
        s = stats[0]
        assert s["instance_id"] == 1
        assert s["area_px"] == 600
        assert s["bbox_x"] == 30
        assert s["bbox_y"] == 20
        assert s["bbox_w"] == 30
        assert s["bbox_h"] == 20
        assert s["area_pct"] == pytest.approx(600 / 10000 * 100, rel=1e-3)

    def test_two_instances(self):
        """Two instances return two stats entries."""
        instance_map = np.zeros((100, 100), dtype=np.uint16)
        instance_map[10:30, 10:30] = 1  # blob 1
        instance_map[60:80, 60:80] = 2  # blob 2
        stats = compute_instance_stats(instance_map, (100, 100))
        assert len(stats) == 2
        ids = [s["instance_id"] for s in stats]
        assert 1 in ids
        assert 2 in ids

    def test_area_pct_with_image_shape(self):
        """area_pct is relative to image_shape, not instance_map."""
        instance_map = np.zeros((50, 50), dtype=np.uint16)
        instance_map[0:25, 0:25] = 1  # 625 px
        stats = compute_instance_stats(instance_map, image_shape=(100, 100))
        assert len(stats) == 1
        # 625 / (100*100) * 100 = 6.25%
        assert stats[0]["area_pct"] == pytest.approx(6.25, rel=1e-3)


# ============================================================
# per_instance_dice
# ============================================================

class TestPerInstanceDice:
    """Tests for per_instance_dice()."""

    def test_identical_maps(self):
        """Two identical maps give Dice=1.0."""
        instance_map = np.zeros((100, 100), dtype=np.uint16)
        instance_map[10:30, 10:30] = 1
        instance_map[60:80, 60:80] = 2
        result = per_instance_dice(instance_map, instance_map)
        assert result["mean_dice"] == pytest.approx(1.0)
        assert result["matched_pairs"] == 2

    def test_both_empty(self):
        """Both empty maps give Dice=1.0."""
        empty = np.zeros((100, 100), dtype=np.uint16)
        result = per_instance_dice(empty, empty)
        assert result["mean_dice"] == pytest.approx(1.0)
        assert result["matched_pairs"] == 0

    def test_one_empty(self):
        """One empty, one non-empty gives Dice=0.0."""
        empty = np.zeros((100, 100), dtype=np.uint16)
        non_empty = np.zeros((100, 100), dtype=np.uint16)
        non_empty[10:30, 10:30] = 1
        result = per_instance_dice(non_empty, empty)
        assert result["mean_dice"] == pytest.approx(0.0)
        result2 = per_instance_dice(empty, non_empty)
        assert result2["mean_dice"] == pytest.approx(0.0)

    def test_disjoint_maps(self):
        """Non-overlapping instances give Dice=0.0."""
        pred = np.zeros((100, 100), dtype=np.uint16)
        pred[10:30, 10:30] = 1
        gt = np.zeros((100, 100), dtype=np.uint16)
        gt[60:80, 60:80] = 1
        result = per_instance_dice(pred, gt)
        assert result["mean_dice"] == pytest.approx(0.0)
