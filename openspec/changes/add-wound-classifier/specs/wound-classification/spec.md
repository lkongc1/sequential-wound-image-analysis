# Wound Classification Specification

## Purpose

Classify segmented wound regions into 7 trauma types using EfficientNet-B3 with 4-channel masked-crop input.

## Requirements

### Requirement: ClassificationDataset

The system MUST provide a PyTorch Dataset that loads images from `data-clasificador/` split CSVs, applies albumentations transforms, and returns a 4-channel tensor (RGB + binary mask) with the integer class label. When `use_mask=False`, it SHALL return 3-channel RGB only. The mask channel SHALL be generated from a provided segmentation mask file path or set to all-zeros when unavailable.

#### Scenario: 4-channel mode with valid mask

- GIVEN a train CSV row with `image_path` and `mask_path` columns
- WHEN `ClassificationDataset.__getitem__` is called with `use_mask=True`
- THEN it returns a (4, H, W) tensor and an int label in [0..6]

#### Scenario: Mask file missing

- GIVEN a train CSV row where `mask_path` does not exist on disk
- WHEN the dataset loads that sample
- THEN it SHALL raise `FileNotFoundError` with the missing path

### Requirement: WoundClassifier Model

The system MUST provide an EfficientNet-B3 wrapper via `timm` with: (a) first conv layer expanded to 4 input channels — RGB weights copied from pretrained, mask channel initialized to zeros; (b) classification head replaced with `nn.Linear(1536, 7)` + softmax. The model MUST accept `pretrained=True` (default) to load ImageNet weights.

#### Scenario: 4-channel forward pass

- GIVEN a `WoundClassifier` instance with `pretrained=True`
- WHEN a (B, 4, 384, 384) tensor is passed
- THEN it outputs (B, 7) log-probabilities
- AND no shape mismatch error occurs

#### Scenario: Pretrained weights absent

- GIVEN `pretrained=False`
- WHEN the model is instantiated
- THEN all weights are randomly initialized
- AND the first conv layer accepts 4 channels

### Requirement: Classifier Training

The system MUST provide a PyTorch Lightning module with: class-weighted `CrossEntropyLoss`, AdamW optimizer at 1e-4 with cosine annealing, albumentations augmentation (RandomResizedCrop, HorizontalFlip, ColorJitter, RandomRotation), and early stopping on validation macro-F1 with patience=10. The module SHALL log per-class F1 to TensorBoard.

#### Scenario: Full training run

- GIVEN balanced train/val splits in `data-clasificador/`
- WHEN `train_classifier.py` runs with default config
- THEN training completes within max_epochs
- AND best checkpoint is saved based on val macro-F1
- AND per-class F1 and confusion matrix are logged

#### Scenario: Early stopping triggers

- GIVEN val F1 does not improve for 10 consecutive epochs
- WHEN training reaches epoch N
- THEN training SHALL stop at epoch N
- AND the best checkpoint (epoch N-10) is retained

### Requirement: Model Registry Integration

The system MUST register the classifier factory via `@register_model("wound_classifier")` in `src/models/factory.py`. Importing the classifier module SHALL trigger the decorator side-effect. Calling `create_model("wound_classifier")` MUST return a configured `WoundClassifier` without requiring a separate import.

#### Scenario: Model loads from registry

- GIVEN `from src.models import factory`
- WHEN `create_model("wound_classifier", num_classes=7, pretrained=False)` is called
- THEN a `WoundClassifier` instance is returned
- AND `create_model("wound_classifier")` works without explicit imports

### Requirement: ClassificationConfig

The system MUST provide a `ClassificationConfig` dataclass in `src/config.py` with fields: `num_classes` (7), `image_size` (384, 384), `checkpoint_path`, `confidence_threshold` (0.5), `top_k` (3), `class_names` (list of 7 Spanish labels), `use_mask` (True), and `batch_size`. Path fields SHALL resolve to absolute in `__post_init__`.

#### Scenario: Config instantiation with defaults

- GIVEN `ClassificationConfig()` with no arguments
- WHEN instantiated
- THEN `num_classes=7`, `confidence_threshold=0.5`, `top_k=3`
- AND `class_names` is `["raspón","hematoma","quemadura","corte","laceración","punción","piel_sana"]`
