# Design: Add Wound Type Classifier

## Technical Approach

Extend the YOLO→U-Net segmentation pipeline with a post-segmentation EfficientNet-B3 classifier. The model takes 4-channel input (RGB + binary mask) via an expanded `conv_stem`, outputs softmax over 7 Spanish wound types. Training uses PyTorch Lightning with class-weighted loss and RandAugment. Inference integrates via `--tipo CHECKPOINT` flag in existing CLI scripts.

```
Image ─► YOLO(bbox) ─► Crop ─► U-Net(mask) ─► MaskedCrop(4ch) ─► Classifier ─► wound_type + confidence
                                                                       ▲
Roboflow ─► LabelNorm ─► Balance(~600/cls) ─► 70/15/15 Split ─► Train ─┘
```

## Architecture Decisions

| Decision | Choice | Rejected | Rationale |
|---|---|---|---|
| 4th channel init | Copy green-channel weights | Zero-init | Green channel most relevant for medical images; zero-init would produce dead weights during training (4th channel input is zero, no gradient flows). Copying ensures mask contributes meaningful signal at inference. |
| Training framework | PyTorch Lightning | Custom Trainer (existing) | Classification has different needs: built-in LR scheduling, early-stopping on F1, TensorBoard logging. Existing Trainer is segmentation-specific (Dice/IoU). No value in adapting it. |
| Augmentation for small data | RandAugment in Albumentations + MixUp/CutMix in training_step | Heavy spatial transforms | ~600/class needs strong regularization. RandAugment (Albumentations `RandAugment` wrapper) handles color/spatial. MixUp/CutMix are batch-level — implemented in `training_step`, not dataset transforms. |
| Model registry pattern | `@register_model("wound_classifier")` in new `src/models/classifier.py`, imported from `factory.py` | Inline in factory.py | Follows existing pattern (`unet.py`, `attention_unet.py`). Factory only gains 1 import line. |
| Data split storage | Split CSVs (`train.csv`, `val.csv`, `test.csv`) with `image_path,label` | Subdirectory structure | Matches existing `dataset_final.csv` pattern. Subdirectory structure is created alongside for direct browsing. |
| Abstention logic | `max_conf < 0.5 → "desconocido"`, `≥ 0.5 → class_name` | Top-K rejection | Simple, interpretable. Threshold is configurable via `ClassificationConfig.confidence_threshold`. Meets spec: confidence exactly at 0.5 returns class name. |
| Zero detections | Write `SIN_DETECCION` to `_tipo.txt`, skip classifier forward pass | Classify full image | Avoids misleading classification on empty/wrong-region input. YOLO is the gatekeeper — no detections means nothing to classify. |

## Data Flow

```
┌─ DOWNLOAD ──────────────────────────────────────────────────────────┐
│ Roboflow SDK → data-clasificador/raw/{dataset}/                      │
│   SmartHeal (1.5K), BasicWoundClassify (2K), BurnWound (3.6K)        │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─ NORMALIZE ──────────────────────────────────────────────────────────┐
│ Label map: {abrasion→raspón, bruise→hematoma, burn→quemadura,        │
│   cut→corte, laceration→laceración, stab→punción, normal_skin→piel_sana}│
│ Unknown label → FAIL with error listing unknown label                 │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─ BALANCE ────────────────────────────────────────────────────────────┐
│ Target: ~600/class. Random undersample majority classes (seed=42).   │
│ Classes below target kept as-is. Log final per-class counts.          │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─ SPLIT ──────────────────────────────────────────────────────────────┐
│ Stratified 70/15/15 (sklearn train_test_split, random_state=42).     │
│ Output: data-clasificador/{train,val,test}/{class_name}/ + CSVs.     │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─ TRAIN ──────────────────────────────────────────────────────────────┐
│ Lightning: AdamW(1e-4), cosine annealing, class-weighted CE loss.    │
│ EarlyStopping on val macro-F1 (patience=10). Save best ckpt by F1.   │
│ Metrics: accuracy, per-class F1, confusion matrix → TensorBoard.     │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─ INFERENCE ──────────────────────────────────────────────────────────┐
│ --tipo models/classifier/best.pth  added to predecir.py + yolo_unet  │
│ Per-instance: crop→maskedCrop(4ch)→classify → _tipo.txt              │
│ Format: "instance_id: class_name (confidence)" / "SIN_DETECCION"     │
└──────────────────────────────────────────────────────────────────────┘
```

## File Changes

| File | Action | Description |
|---|---|---|
| `src/models/classifier.py` | **Create** | `WoundClassifier(nn.Module)` — EfficientNet-B3 wrapper, 4ch conv_stem, 7-class head. `@register_model("wound_classifier")`. |
| `src/datasets/classification_dataset.py` | **Create** | `ClassificationDataset(Dataset)` — loads from split CSV, returns (C,384,384) tensor + int label. `use_mask` flag controls channels. |
| `src/config.py` | **Modify** | Add `ClassificationConfig` dataclass (num_classes, image_size, confidence_threshold, class_names, etc.) |
| `src/models/factory.py` | **Modify** | Add `from src.models import classifier` (1 line at bottom) |
| `scripts/download_classification_data.py` | **Create** | Roboflow download → normalize → balance → split. |
| `scripts/train_classifier.py` | **Create** | Lightning training script: dataset → model → trainer → checkpoint. |
| `scripts/inference/predecir.py` | **Modify** | Add `--tipo CHECKPOINT` argument. After segmentation, classify instances. Output `_tipo.txt`. |
| `scripts/inference/predecir_yolo_unet.py` | **Modify** | Add `--tipo CHECKPOINT` argument. Per-bbox classification. Output `_tipo.txt`. |
| `data-clasificador/` | **Create** | Directory structure: `raw/`, `train/`, `val/`, `test/` + CSVs |

## Interfaces / Contracts

```python
# src/models/classifier.py
class WoundClassifier(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True, freeze_backbone: bool = False)
    def forward(self, x: Tensor) -> Tensor  # (B,4,384,384) → (B,7) log-probabilities

# Register: @register_model("wound_classifier")
create_model("wound_classifier", num_classes=7, pretrained=True) → WoundClassifier

# src/datasets/classification_dataset.py
class ClassificationDataset(Dataset):
    def __init__(self, csv_path: Path, class_names: list[str], image_size=(384,384),
                 use_mask: bool = False, augment: bool = False)
    def __getitem__(self, idx) -> tuple[Tensor, int]  # (C,H,W), label_index

# src/config.py — new dataclass
@dataclass
class ClassificationConfig:
    num_classes: int = 7
    model_name: str = "efficientnet_b3"
    image_size: tuple[int, int] = (384, 384)
    batch_size: int = 16
    learning_rate: float = 1e-4
    class_names: list[str] = field(default_factory=lambda: [
        "raspón","hematoma","quemadura","corte","laceración","punción","piel_sana"])
    confidence_threshold: float = 0.5
    top_k: int = 3
    use_mask: bool = True
    checkpoint_path: Path | None = None
    max_epochs: int = 50
    patience: int = 10

# Inference integration
# predecir.py gains:
parser.add_argument("--tipo", type=Path, help="Classifier checkpoint for wound type")
# Output: {stem}_tipo.txt — "instance_id: class_name (confidence)" per line or "SIN_DETECCION"
```

## Error Handling

| Scenario | Behavior |
|---|---|
| Unknown Roboflow label | Download script fails with error listing unknown label and suggestion to update mapping table |
| Mask file missing in dataset | `ClassificationDataset.__getitem__` raises `FileNotFoundError` with full path |
| Classifier checkpoint not found | CLI prints error and exits before loading U-Net |
| YOLO returns zero detections | Classifier skipped entirely; `_tipo.txt` written as `SIN_DETECCION` |
| U-Net produces all-zero mask | Classifier receives zero 4th channel — operates on RGB only, valid |
| CUDA OOM at inference | Fall back to CPU with warning printed |
| Corrupted image in dataset | `load_image_safely` (shared from wound_dataset) raises `FileNotFoundError` |

## Testing Strategy

| Layer | What | How |
|---|---|---|
| Unit | `WoundClassifier` forward pass shape | `torch.randn(2,4,384,384)` → assert `(2,7)` |
| Unit | `ClassificationDataset` modes | 3ch vs 4ch output, label encoding consistency |
| Unit | Label normalization mapping | All 7 known labels map correctly; unknown raises |
| Integration | `create_model("wound_classifier")` | Registry lookup works, returns `WoundClassifier` |
| Integration | Training loop runs 1 epoch | Smoke test: loss decreases, no NaN |
| E2E | `predecir.py --tipo` on sample image | Produces valid `_tipo.txt` with expected format |
| E2E | Abstention threshold | Low-confidence crop → output contains `desconocido` |

## Open Questions

- [ ] Medetec dataset priority: proceed without or pause if behind paywall? (Proposal: proceed without)
- [ ] Fine-tuning on 4ch after initial 3ch training: should we generate pseudo-masks for Roboflow images using existing U-Net to properly train 4th-channel weights? (Recommended as future optimization)
