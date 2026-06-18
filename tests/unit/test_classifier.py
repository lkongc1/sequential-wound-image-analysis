"""Unit tests for WoundClassifier model and ClassificationDataset.

Run with: pytest tests/unit/test_classifier.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

# Setup project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import create_model
from src.models.classifier import WoundClassifier
from src.datasets.classification_dataset import ClassificationDataset
from src.config import ClassificationConfig


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def synthetic_classification_csv(tmp_path: Path) -> Path:
    """Create a synthetic CSV with 3-channel classification data (no masks)."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    records = []
    class_names = [
        "raspón", "hematoma", "quemadura", "corte",
        "laceración", "punción", "piel_sana",
    ]
    rng = np.random.default_rng(seed=42)
    img_idx = 0
    for label in class_names:
        # Create 2 images per class
        for j in range(2):
            img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
            img_path = images_dir / f"img_{img_idx:03d}.png"
            import cv2
            cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            records.append({"image_path": str(img_path), "label": label})
            img_idx += 1
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def synthetic_classification_csv_with_masks(tmp_path: Path) -> Path:
    """Create a synthetic CSV with mask paths for 4-channel testing."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    records = []
    class_names = [
        "raspón", "hematoma", "quemadura", "corte",
        "laceración", "punción", "piel_sana",
    ]
    rng = np.random.default_rng(seed=42)
    img_idx = 0
    for label in class_names:
        for j in range(2):
            import cv2
            img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
            mask = rng.integers(0, 2, (128, 128), dtype=np.uint8) * 255
            img_path = images_dir / f"img_{img_idx:03d}.png"
            mask_path = masks_dir / f"mask_{img_idx:03d}.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mask_path), mask)
            records.append({
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "label": label,
            })
            img_idx += 1
    csv_path = tmp_path / "train_masks.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return csv_path


# ------------------------------------------------------------------ #
# WoundClassifier tests
# ------------------------------------------------------------------ #


class TestWoundClassifier:
    """Tests for the WoundClassifier model."""

    def test_forward_shape_4ch(self):
        """Forward pass with 4-channel input should output (B, 7)."""
        model = WoundClassifier(num_classes=7, pretrained=False)
        model.eval()
        x = torch.randn(2, 4, 384, 384)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 7), f"Expected (2, 7), got {out.shape}"

    def test_forward_shape_batch_8(self):
        """Forward pass should work with different batch sizes."""
        model = WoundClassifier(num_classes=7, pretrained=False)
        model.eval()
        x = torch.randn(8, 4, 384, 384)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 7)

    def test_num_classes_custom(self):
        """Should accept custom num_classes."""
        model = WoundClassifier(num_classes=5, pretrained=False)
        model.eval()
        x = torch.randn(1, 4, 384, 384)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)

    def test_pretrained_flag_accepted(self):
        """pretrained=True should not crash (weights load from timm)."""
        try:
            import timm  # noqa: F401
        except ImportError:
            pytest.skip("timm not installed")
        model = WoundClassifier(num_classes=7, pretrained=True)
        assert isinstance(model, nn.Module)

    def test_first_conv_has_4_input_channels(self):
        """conv_stem should accept 4 input channels after expansion."""
        model = WoundClassifier(num_classes=7, pretrained=False)
        conv_stem = model.backbone.conv_stem  # type: ignore[attr-defined]
        assert conv_stem.in_channels == 4, (
            f"Expected 4 input channels, got {conv_stem.in_channels}"
        )

    def test_green_channel_copy(self):
        """Mask channel (index 3) weights should equal green channel (index 1)."""
        model = WoundClassifier(num_classes=7, pretrained=False)
        weight = model.backbone.conv_stem.weight.data  # type: ignore[attr-defined]
        # Channel 3 (mask) should be identical to channel 1 (green)
        assert torch.equal(weight[:, 3, :, :], weight[:, 1, :, :]), (
            "Mask channel weights should be copied from green channel"
        )

    def test_head_output_classes(self):
        """Classification head should output num_classes features."""
        model = WoundClassifier(num_classes=7, pretrained=False)
        assert model.head[-1].out_features == 7

    def test_freeze_backbone(self):
        """freeze_backbone=True should set backbone params to not require grad."""
        model = WoundClassifier(num_classes=7, pretrained=False, freeze_backbone=True)
        backbone_params = list(model.backbone.parameters())
        assert all(not p.requires_grad for p in backbone_params), (
            "All backbone params should be frozen"
        )
        # Head params should still be trainable
        head_params = list(model.head.parameters())
        assert all(p.requires_grad for p in head_params), (
            "Head params should remain trainable"
        )

    def test_output_is_log_probabilities(self):
        """Output should be log-probabilities (log_softmax)."""
        model = WoundClassifier(num_classes=7, pretrained=False)
        model.eval()
        x = torch.randn(1, 4, 384, 384)
        with torch.no_grad():
            out = model(x)
        # log_softmax output sums to < 0 (log of < 1), values are negative
        assert torch.all(out <= 0), "Log-probabilities should be ≤ 0"
        # exp sum should be close to 1
        probs = torch.exp(out)
        assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5)


# ------------------------------------------------------------------ #
# Model registry tests
# ------------------------------------------------------------------ #


class TestClassifierRegistry:
    """Tests for the wound_classifier registry integration."""

    def test_create_model_wound_classifier_returns_woundclassifier(self):
        """create_model('wound_classifier') should return WoundClassifier."""
        model = create_model("wound_classifier", num_classes=7, pretrained=False)
        assert isinstance(model, WoundClassifier), (
            f"Expected WoundClassifier, got {type(model)}"
        )

    def test_create_model_with_custom_num_classes(self):
        """Should forward kwargs to WoundClassifier constructor."""
        model = create_model("wound_classifier", num_classes=3, pretrained=False)
        assert model.num_classes == 3

    def test_wound_classifier_registered_in_registry(self):
        """'wound_classifier' should be in MODEL_REGISTRY."""
        from src.models.factory import MODEL_REGISTRY
        assert "wound_classifier" in MODEL_REGISTRY


# ------------------------------------------------------------------ #
# ClassificationDataset tests
# ------------------------------------------------------------------ #


class TestClassificationDataset:
    """Tests for the ClassificationDataset."""

    CLASS_NAMES = [
        "raspón", "hematoma", "quemadura", "corte",
        "laceración", "punción", "piel_sana",
    ]

    def test_dataset_3ch_mode(self, synthetic_classification_csv: Path):
        """use_mask=False should return (3, H, W) tensors."""
        ds = ClassificationDataset(
            csv_path=synthetic_classification_csv,
            class_names=self.CLASS_NAMES,
            image_size=(384, 384),
            use_mask=False,
            augment=False,
        )
        tensor, label = ds[0]
        assert tensor.shape == (3, 384, 384), (
            f"Expected (3, 384, 384), got {tensor.shape}"
        )
        assert isinstance(label, int)
        assert 0 <= label < 7

    def test_dataset_4ch_mode(self, synthetic_classification_csv_with_masks: Path):
        """use_mask=True should return (4, H, W) tensors."""
        ds = ClassificationDataset(
            csv_path=synthetic_classification_csv_with_masks,
            class_names=self.CLASS_NAMES,
            image_size=(384, 384),
            use_mask=True,
            augment=False,
        )
        tensor, label = ds[0]
        assert tensor.shape == (4, 384, 384), (
            f"Expected (4, 384, 384), got {tensor.shape}"
        )
        assert isinstance(label, int)
        assert 0 <= label < 7

    def test_dataset_len(self, synthetic_classification_csv: Path):
        """__len__ should match number of rows in CSV."""
        ds = ClassificationDataset(
            csv_path=synthetic_classification_csv,
            class_names=self.CLASS_NAMES,
            image_size=(384, 384),
            use_mask=False,
        )
        df = pd.read_csv(synthetic_classification_csv)
        assert len(ds) == len(df)

    def test_dataset_labels_consistent(self, synthetic_classification_csv: Path):
        """Label indices should match class_names order."""
        ds = ClassificationDataset(
            csv_path=synthetic_classification_csv,
            class_names=self.CLASS_NAMES,
            image_size=(384, 384),
            use_mask=False,
        )
        df = pd.read_csv(synthetic_classification_csv)
        for i in range(len(ds)):
            _, label_idx = ds[i]
            expected_label = df.iloc[i]["label"]
            expected_idx = self.CLASS_NAMES.index(expected_label)
            assert label_idx == expected_idx, (
                f"Sample {i}: expected idx {expected_idx} for '{expected_label}', "
                f"got {label_idx}"
            )

    def test_dataset_with_augment(self, synthetic_classification_csv: Path):
        """augment=True should still produce valid shapes."""
        ds = ClassificationDataset(
            csv_path=synthetic_classification_csv,
            class_names=self.CLASS_NAMES,
            image_size=(384, 384),
            use_mask=False,
            augment=True,
        )
        tensor, label = ds[0]
        assert tensor.shape == (3, 384, 384)

    def test_dataset_4ch_no_mask_column_uses_zeros(self, synthetic_classification_csv: Path):
        """use_mask=True without mask_path column should use all-zeros mask."""
        ds = ClassificationDataset(
            csv_path=synthetic_classification_csv,
            class_names=self.CLASS_NAMES,
            image_size=(384, 384),
            use_mask=True,
        )
        tensor, _ = ds[0]
        assert tensor.shape == (4, 384, 384)
        # The 4th channel should be all zeros (or normalization equivalent)
        mask_channel = tensor[3, :, :]
        # After normalization with mean=0.485/std=0.229: zeros → (0 - 0.485)/0.229 ≈ -2.12
        # Just verify it's uniform (all same value since input was zeros)
        assert torch.allclose(mask_channel, mask_channel[0, 0].expand_as(mask_channel)), (
            "Mask channel from all-zeros input should be uniform after normalization"
        )

    def test_dataset_raises_on_missing_mask_file(self, synthetic_classification_csv_with_masks: Path, tmp_path: Path):
        """Should raise FileNotFoundError when mask_path points to missing file."""
        # Append a row with a non-existent mask path
        df = pd.read_csv(synthetic_classification_csv_with_masks)
        bad_row = df.iloc[0].copy()
        bad_row["mask_path"] = str(tmp_path / "nonexistent_mask.png")
        bad_row["image_path"] = df.iloc[0]["image_path"]  # reuse existing image
        df = pd.concat([df, pd.DataFrame([bad_row])], ignore_index=True)
        bad_csv = tmp_path / "bad_masks.csv"
        df.to_csv(bad_csv, index=False)

        ds = ClassificationDataset(
            csv_path=bad_csv,
            class_names=self.CLASS_NAMES,
            image_size=(384, 384),
            use_mask=True,
        )
        # The bad row is the last one
        with pytest.raises(FileNotFoundError, match="nonexistent_mask"):
            _ = ds[len(ds) - 1]

    def test_dataset_raises_on_unknown_label(self, tmp_path: Path):
        """Should raise ValueError when CSV contains a label not in class_names."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        import cv2
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img_path = images_dir / "test.png"
        cv2.imwrite(str(img_path), img)

        df = pd.DataFrame([{"image_path": str(img_path), "label": "fractura"}])
        csv_path = tmp_path / "bad_labels.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="fractura"):
            ClassificationDataset(
                csv_path=csv_path,
                class_names=self.CLASS_NAMES,
                image_size=(384, 384),
            )


# ------------------------------------------------------------------ #
# ClassificationConfig tests
# ------------------------------------------------------------------ #


class TestClassificationConfig:
    """Tests for the ClassificationConfig dataclass."""

    def test_default_values(self):
        """Default instantiation should match design contract."""
        cfg = ClassificationConfig()
        assert cfg.num_classes == 7
        assert cfg.confidence_threshold == 0.5
        assert cfg.top_k == 3
        assert cfg.use_mask is True
        assert cfg.batch_size == 16
        assert cfg.learning_rate == 1e-4
        assert len(cfg.class_names) == 7
        assert cfg.class_names[0] == "raspón"
        assert cfg.class_names[-1] == "piel_sana"

    def test_paths_resolved_to_absolute(self):
        """__post_init__ should resolve relative paths to absolute."""
        cfg = ClassificationConfig()
        assert cfg.train_csv.is_absolute(), "train_csv should be absolute"
        assert cfg.val_csv.is_absolute(), "val_csv should be absolute"
        assert cfg.test_csv.is_absolute(), "test_csv should be absolute"
        assert cfg.output_dir.is_absolute(), "output_dir should be absolute"

    def test_checkpoint_path_none_by_default(self):
        """checkpoint_path should be None when not provided."""
        cfg = ClassificationConfig()
        assert cfg.checkpoint_path is None

    def test_checkpoint_path_resolved_if_provided(self):
        """checkpoint_path should be resolved when a relative path is given."""
        cfg = ClassificationConfig(checkpoint_path=Path("models/classifier/best.pth"))
        assert cfg.checkpoint_path.is_absolute()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
