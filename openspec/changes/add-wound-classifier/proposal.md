# Proposal: Add Wound Type Classifier

**Status**: proposed  
**Created**: 2026-06-15  
**Taxonomy**: 7-class wound type (raspón, hematoma, quemadura, corte, laceración, punción, piel_sana)

## Intent

The system segments wounds (WHERE) but cannot classify them (WHAT TYPE). A clinician needs to distinguish a burn from a laceration to determine treatment. This adds a post-segmentation EfficientNet-B3 classifier to the YOLO→U-Net pipeline, outputting a wound type label + confidence per detected wound region.

## Scope

### In Scope
- 7-class wound classifier with EfficientNet-B3 backbone (~384×384 input, 4-channel masked crop)
- Data pipeline: download 3 Roboflow datasets into `data-clasificador/`, normalize English→Spanish labels, balance to ~600/class (~4200 total)
- Training script with albumentations augmentation, class-weighted CrossEntropyLoss, early stopping
- `@register_model("wound_classifier")` in factory.py
- `--tipo` flag in `predecir.py` and `predecir_yolo_unet.py` for CLI inference
- Abstention threshold 0.5: return "desconocido" below confidence
- Masked-crop input format: background black, only wound visible (mask as attention channel)

### Out of Scope
- Chronic wound types (diabetic ulcers, pressure ulcers, venous ulcers)
- Real-time optimization (TensorRT, quantization, ONNX export) — deferred
- Multi-label classification (one wound = one type)
- Wound severity grading (depth, stage, infection)
- Full API `/diagnosis` endpoint wiring — deferred to follow-up change
- Medetec data integration — optional secondary source, attempted only if easily available

## Capabilities

### New Capabilities
- `wound-classification`: Classify segmented wound regions into 7 trauma types. Input: masked crop (4-channel). Output: type label + confidence score. Abstention below 0.5 confidence.
- `classification-data-pipeline`: Download, unify, balance, and split classification datasets from Roboflow into train/val/test splits in `data-clasificador/`.

### Modified Capabilities
- `wound-segmentation-inference`: Extended CLI (`--tipo` flag) to run classifier after segmentation; outputs type label appended to mask/stats output. Existing behavior unchanged when flag is absent.

## Approach

**Architecture**: Independent classifier post-segmentation (Option A from exploration).
```
Image → YOLO (bbox) → Crop → U-Net (mask) → MaskedCrop(4ch) → Classifier → wound_type
```

**Model**: EfficientNet-B3 (via `timm`) with ImageNet pretrained weights. Replace classification head: `nn.Linear(1536, 7)`. First conv layer expanded to accept 4 channels (RGB + mask). Output: softmax over 7 classes.

**Data**: Three Roboflow datasets downloaded via API, label-mapped from English to Spanish taxonomy, undersampled to the minority class (~600/class), split 70/15/15 train/val/test.

**Integration**: CLI-first. `--tipo` flag triggers classifier loading + inference on each YOLO-cropped region after U-Net mask generation. Output: `{image_stem}_tipo.txt` with per-wound (type, confidence) pairs. API wiring deferred.

**Training**: PyTorch Lightning `LightningModule` wrapper. Augmentation: RandomResizedCrop, HorizontalFlip, ColorJitter, RandomRotation. Loss: `CrossEntropyLoss(weight=class_weights)`. Optimizer: AdamW, 1e-4, cosine annealing. Early stopping on val F1 (patience=10). Target: ~30 min on single GPU.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/models/classifier.py` | New | EfficientNet-B3 wrapper with `@register_model` |
| `src/models/factory.py` | Modified | Import classifier module to trigger registration |
| `src/config.py` | Modified | Add `ClassificationConfig` dataclass |
| `src/datasets/classification_dataset.py` | New | Dataset class for 4-channel masked crops |
| `scripts/download_classification_data.py` | New | Roboflow download + label normalization |
| `scripts/train_classifier.py` | New | PyTorch Lightning training script |
| `scripts/inference/predecir.py` | Modified | Add `--tipo` flag + classifier inference path |
| `scripts/inference/predecir_yolo_unet.py` | Modified | Add `--tipo` flag + classifier inference path |
| `data-clasificador/` | New | Classification dataset directory (separate from `data/`) |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Domain shift: classifier trained on full-frame Roboflow images, evaluated on YOLO-cropped pipeline images | High | Use masked-crop input format (4ch: RGB+mask) to normalize representation; evaluate on real pipeline crops post-training |
| Class imbalance in Roboflow sources (e.g., burns 3691 images vs stab ~100) | High | Undersample to ~600/class; use class-weighted loss; monitor per-class F1, not just accuracy |
| Label inconsistency across datasets (e.g., "abrasion" vs "abrasions") | Medium | Normalize via explicit mapping table; validate all labels post-download |
| Roboflow API rate limits or dataset unavailability | Low | Cache downloads locally; fallback to manual download instructions |
| 4-channel input not supported by timm's default first conv layer | Medium | Replace `conv_stem` with a new Conv2d(4, 40, ...); copy RGB weights, init mask channel with zeros |

## Rollback Plan

1. Remove `--tipo` flag handling from inference scripts (revert to pre-change behavior).
2. Remove `from src.models import classifier` import from factory.py.
3. Delete `data-clasificador/` and model checkpoint.
4. No database migrations, no API contract changes — pure additive with no side effects on segmentation pipeline.

## Dependencies

- `timm` (already in requirements.txt for EfficientNet encoders)
- `roboflow` Python SDK (new, for dataset download)
- PyTorch Lightning (already used in training scripts)
- `albumentations` (already used for segmentation augmentation)
- Valid Roboflow API key: `DVnQXr1udAOhcikIJWnJ` (private key)

## Success Criteria

- [ ] Per-class F1 ≥ 0.70 on held-out test set for all 7 classes
- [ ] Overall accuracy ≥ 0.75 on balanced 7-class test set
- [ ] Abstention: confidence < 0.5 returns "desconocido" (no forced classification)
- [ ] `--tipo` flag works in both `predecir.py` and `predecir_yolo_unet.py` without breaking existing behavior
- [ ] `@register_model("wound_classifier")` loads via `create_model("wound_classifier")` without errors
- [ ] Confusion matrix shows no class confused > 30% with any other single class

## Proposal Question Round

These questions aim to refine edge cases before spec/design. Answer, skip, or ask for a second round.

1. **Empty crops**: When YOLO detects no wound, the pipeline passes the full image to U-Net. Should the classifier run on the full U-Net mask or skip classification entirely?

2. **Multi-wound images**: If an image has 3 wounds (e.g. laceration + bruise + abrasion), should we output one label per wound or one aggregate label per image?

3. **Confidence output format**: Should the output be `"laceración (0.87)"` or `[("laceración", 0.87), ("corte", 0.06), ...]` (top-K)? The latter helps clinicians see alternative diagnoses.

4. **Medetec priority**: If Medetec data is behind a paywall or requires manual approval, should we proceed without it or pause until it's obtained?
