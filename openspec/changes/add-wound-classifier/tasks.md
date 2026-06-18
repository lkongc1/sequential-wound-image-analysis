# Tasks: Add Wound Type Classifier

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~630 (490 new + 141 modified) |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | PR 1: Foundation (~370 lines) → PR 2: Training+Inference (~260 lines) |
| Delivery strategy | ask-on-risk |
| Chain strategy | pending |

Decision needed before apply: Yes
Chained PRs recommended: Yes
Chain strategy: pending
400-line budget risk: High

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | Data pipeline + model core | PR 1 | Self-contained: download script runs independently, model unit-testable with randn |
| 2 | Training script + CLI integration | PR 2 | Base: PR 1 branch. Depends on classifier.py + classification_dataset.py |

## Phase 1: Data Pipeline (foundation)

- [x] 1.1 Create `scripts/download_classification_data.py` — Roboflow SDK download (3 datasets, API key `DVnQXr1udAOhcikIJWnJ`) into `data-clasificador/raw/`, skip if cached. Write manifest CSV.
- [x] 1.2 Implement label normalization in download script — mapping table (English→Spanish, 7 classes), fail with descriptive error on unknown labels.
- [x] 1.3 Implement class balancing in download script — random undersample to ~600/class (seed=42), log per-class counts.
- [x] 1.4 Implement stratified 70/15/15 split — `sklearn.train_test_split` (random_state=42), write `data-clasificador/{train,val,test}.csv` with `image_path,label` columns.
- [x] 1.5 Verify — run script, check manifest CSV, per-class counts in each split, label consistency across all 3 sources.

## Phase 2: Model + Dataset (core)

- [x] 2.1 Add `ClassificationConfig` dataclass to `src/config.py` — fields per design contract (num_classes, image_size, confidence_threshold, class_names, top_k, use_mask, batch_size, etc.), `__post_init__` resolves relative paths.
- [x] 2.2 Create `src/datasets/classification_dataset.py` — `ClassificationDataset(Dataset)`, loads from split CSV, `use_mask` flag controls 3ch vs 4ch output (mask channel from `mask_path` column or all-zeros), augment flag enables RandAugment transforms.
- [x] 2.3 Create `src/models/classifier.py` — `WoundClassifier(nn.Module)`, EfficientNet-B3 via timm, expand `conv_stem` to 4ch (copy green-channel weights for mask channel), replace head with `nn.Linear(1536,7)`, `@register_model("wound_classifier")`.
- [x] 2.4 Wire registry — add `from src.models import classifier` at bottom of `src/models/factory.py`.
- [x] 2.5 Unit-test — `torch.randn(2,4,384,384)` → assert `(2,7)`, `create_model("wound_classifier", num_classes=7)` returns WoundClassifier, dataset 3ch vs 4ch modes return correct tensor shapes.

## Phase 3: Training

- [ ] 3.1 Create `scripts/train_classifier.py` — PyTorch Lightning module with class-weighted CrossEntropyLoss, AdamW(1e-4) + cosine annealing, RandAugment transforms, MixUp/CutMix in training_step.
- [ ] 3.2 Add callbacks — EarlyStopping on val macro-F1 (patience=10), ModelCheckpoint (best by val F1), TensorBoard logging with per-class F1 and confusion matrix.
- [ ] 3.3 Smoke-test — run 1 epoch, verify loss decreases, no NaN, checkpoint saved.

## Phase 4: Inference Integration

- [ ] 4.1 Add `--tipo CHECKPOINT` to `scripts/inference/predecir.py` — after watershed/SAM2 instances, load classifier, create 4ch masked crop per instance (RGB+mask), classify, write `{stem}_tipo.txt` with per-instance `id: class (conf)` format. Skip classifier when mask has zero positive pixels → write `SIN_DETECCION`.
- [ ] 4.2 Add `--tipo CHECKPOINT` to `scripts/inference/predecir_yolo_unet.py` — after YOLO→U-Net per-bbox pipeline, classify each bbox crop. Zero YOLO detections → write `SIN_DETECCION`, skip classifier forward pass.
- [ ] 4.3 Implement abstention — `max_conf < confidence_threshold` → output `desconocido`, confidence exactly at 0.5 returns class name. CUDA OOM → fallback to CPU with warning.
- [ ] 4.4 E2E verify — run `predecir.py sample.png --tipo models/classifier/best.pth`, validate `_tipo.txt` format, test low-confidence abstention, test empty-image `SIN_DETECCION`.
