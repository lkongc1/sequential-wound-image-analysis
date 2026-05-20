# System Architecture

## Overview

Wound Segmentation is a medical AI system for automated wound detection and tracking. The system uses U-Net based deep learning models for pixel-wise wound segmentation.

## Components

```
┌─────────────────────────────────────────────────────┐
│                   API Layer (FastAPI)                 │
│  /diagnosis  /patients  /health  /upload             │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Inference Engine                        │
│  Predictor → Postprocessing → Ensemble              │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Model Registry                          │
│  Production models with version tracking            │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Data Pipeline                          │
│  Acquisition → QC → Preprocessing → Augmentation   │
└─────────────────────────────────────────────────────┘
```

## Module Descriptions

### `src/data/`
- `acquisition.py`: Clinical image acquisition protocols
- `preprocessing.py`: Image normalization and illumination correction
- `augmentation.py`: Training-time augmentation with Albumentations
- `quality_control.py`: Image quality checks (resolution, brightness, sharpness)
- `validators.py`: Data integrity and domain shift detection

### `src/datasets/`
- `wound_dataset.py`: PyTorch Dataset implementations
- `split_strategy.py`: Train/validation/test splitting
- `transforms.py`: Albumentations transform compositions

### `src/models/`
- `unet.py`: Lightweight U-Net (8GB VRAM optimized)
- `attention_unet.py`: U-Net with attention gates
- `nested_unet.py`: U-Net++ architecture
- `factory.py`: Model factory for architecture selection

### `src/training/`
- `trainer.py`: Main training loop with mixed precision
- `callbacks.py`: EarlyStopping, ModelCheckpoint
- `lr_schedulers.py`: Cosine annealing with warmup
- `mixed_precision.py`: AMP utilities

### `src/losses/`
- `dice_loss.py`: Dice Loss for class imbalance
- `bce_dice_loss.py`: Combined BCE + Dice
- `tversky_loss.py`: Tversky loss (configurable FN/FP tradeoff)
- `focal_loss.py`: Focal loss for severe imbalance

### `src/metrics/`
- `segmentation.py`: Dice, IoU, Sensitivity, Specificity
- `clinical_metrics.py`: Wound area, color change tracking
- `longitudinal.py`: Healing progression metrics
- `reporting.py`: Clinical report generation

### `src/inference/`
- `predictor.py`: Model inference wrapper
- `postprocessing.py`: Mask cleaning and morphology
- `ensemble.py`: Multi-model ensemble predictions

### `src/api/`
- FastAPI application with routes for diagnosis, patient management, health checks

## Configuration

Environment-specific configs in `config/environments/`:
- `development.yaml`: Local GPU training
- `staging.yaml`: Larger batch sizes, more epochs
- `production.yaml`: Highest resolution, FDA compliance mode

## Deployment

Docker containers for:
- Training (`Dockerfile.training`)
- API service (`Dockerfile.api`)
- Minimal inference (`Dockerfile.inference`)
- Jenkins agent (`Dockerfile.jenkins`)
