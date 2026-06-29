#!/usr/bin/env python3
"""Train wound type classifier with PyTorch Lightning.

Fine-tunes EfficientNet-B3 on 6/7-class wound type data with:
- Class-weighted CrossEntropyLoss + label smoothing
- AdamW(1e-4) + cosine annealing
- RandAugment transforms (N=3, M=9)
- MixUp/CutMix in training_step (alpha=0.4)
- Freeze backbone for first N epochs, then unfreeze
- EarlyStopping on val macro-F1 (patience=10)
- ModelCheckpoint (best by val F1)
- TensorBoard logging: per-class F1, confusion matrix

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --epochs 30 --lr 5e-5
    python scripts/train_classifier.py --csv-prefix data-clasificador
    python scripts/train_classifier.py --dry-run  # 1 epoch smoke test
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ClassificationConfig
from src.datasets.classification_dataset import ClassificationDataset
from src.models.factory import create_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ #
# DataModule
# ------------------------------------------------------------------ #


class WoundDataModule(pl.LightningDataModule):
    """LightningDataModule for wound classification splits.

    Loads from train.csv and val.csv, applies RandAugment (optionally
    with domain adaptation) to training set.
    """

    def __init__(self, config: ClassificationConfig, domain_adapt: bool = True):
        super().__init__()
        self.cfg = config
        self.domain_adapt = domain_adapt
        self.train_dataset: Optional[ClassificationDataset] = None
        self.val_dataset: Optional[ClassificationDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = ClassificationDataset(
                csv_path=self.cfg.train_csv,
                class_names=self.cfg.class_names,
                image_size=self.cfg.image_size,
                use_mask=self.cfg.use_mask,
                augment=True,
                domain_adapt=self.domain_adapt,
            )
            self.val_dataset = ClassificationDataset(
                csv_path=self.cfg.val_csv,
                class_names=self.cfg.class_names,
                image_size=self.cfg.image_size,
                use_mask=self.cfg.use_mask,
                augment=False,
            )
            # Compute class weights from training distribution
            self._compute_class_weights()

    def _compute_class_weights(self) -> None:
        """Compute inverse-frequency class weights for the loss."""
        labels = self.train_dataset.df["label"]  # type: ignore[union-attr]
        counts = labels.value_counts().reindex(self.cfg.class_names, fill_value=0)
        # Inverse frequency weight
        weights = 1.0 / (counts.values + 1e-6)
        weights = weights / weights.sum() * len(self.cfg.class_names)
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
        logger.info(
            "Class weights: %s",
            {cls: f"{w:.3f}" for cls, w in zip(self.cfg.class_names, weights)},
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,  # type: ignore[arg-type]
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,  # Windows multiprocessing safety
            drop_last=True,
            pin_memory=(DEVICE == "cuda"),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,  # type: ignore[arg-type]
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=(DEVICE == "cuda"),
        )


# ------------------------------------------------------------------ #
# LightningModule
# ------------------------------------------------------------------ #


class WoundClassifierLightning(pl.LightningModule):
    """LightningModule wrapping WoundClassifier for training.

    Features:
    - Class-weighted CrossEntropyLoss with label smoothing
    - MixUp and CutMix applied probabilistically in training_step
    - Per-class F1 and confusion matrix logged to TensorBoard
    - Gradual backbone unfreezing after freeze_backbone_epochs
    """

    def __init__(
        self,
        config: ClassificationConfig,
        class_weights: Optional[Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.cfg = config

        self.model = create_model(
            "wound_classifier",
            num_classes=config.num_classes,
            pretrained=True,
            freeze_backbone=config.freeze_backbone,
            dropout=config.dropout,
        )
        self.class_weights = class_weights

        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(DEVICE) if class_weights is not None else None,
            label_smoothing=config.label_smoothing,
        )

        # Metrics
        self._val_preds: list[Tensor] = []
        self._val_targets: list[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        # --- MixUp / CutMix (probabilistic) ---
        if self.cfg.mixup_alpha > 0 and self.cfg.cutmix_alpha > 0:
            if torch.rand(1).item() < 0.5:
                images, labels_a, labels_b, lam = self._mixup(images, labels)
                logits = self.model(images)
                loss = lam * self.criterion(logits, labels_a) + (1 - lam) * self.criterion(logits, labels_b)
            elif torch.rand(1).item() < 0.5:
                images, labels_a, labels_b, lam = self._cutmix(images, labels)
                logits = self.model(images)
                loss = lam * self.criterion(logits, labels_a) + (1 - lam) * self.criterion(logits, labels_b)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
        else:
            logits = self.model(images)
            loss = self.criterion(logits, labels)

        # Compute training accuracy for monitoring
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            if 'labels_a' in dir():
                # Mixed batch — approximate accuracy on primary label
                acc = (preds == labels_a).float().mean()
            else:
                acc = (preds == labels).float().mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def _mixup(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor, float]:
        """Apply MixUp: linearly interpolate two samples."""
        lam = np.random.beta(self.cfg.mixup_alpha, self.cfg.mixup_alpha)
        lam = max(lam, 1 - lam)  # ensure lam >= 0.5 for consistent primary label
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)

        mixed_images = lam * images + (1 - lam) * images[index]
        return mixed_images, labels, labels[index], lam

    def _cutmix(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor, float]:
        """Apply CutMix: replace a random region with another sample."""
        lam = np.random.beta(self.cfg.cutmix_alpha, self.cfg.cutmix_alpha)
        lam = max(lam, 1 - lam)
        batch_size, _, H, W = images.shape
        index = torch.randperm(batch_size, device=images.device)

        # Random bounding box
        cut_ratio = np.sqrt(1 - lam)
        cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
        cy = np.random.randint(0, H)
        cx = np.random.randint(0, W)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, cx + cut_w // 2)

        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        # Adjust lambda to actual area ratio
        lam_adjusted = 1 - ((y2 - y1) * (x2 - x1) / (H * W))
        return mixed_images, labels, labels[index], lam_adjusted

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        logits = self.model(images)
        loss = self.criterion(logits, labels)

        self._val_preds.append(logits.detach().cpu())
        self._val_targets.append(labels.detach().cpu())

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        from torchmetrics.functional import accuracy, f1_score, confusion_matrix

        all_preds = torch.cat(self._val_preds, dim=0)
        all_targets = torch.cat(self._val_targets, dim=0)
        all_pred_labels = torch.argmax(all_preds, dim=1)

        # Macro F1 (main metric)
        macro_f1 = f1_score(
            all_pred_labels, all_targets,
            task="multiclass",
            num_classes=self.cfg.num_classes,
            average="macro",
        )
        self.log("val/macro_f1", macro_f1, on_epoch=True, prog_bar=True)

        # Per-class F1
        per_class_f1 = f1_score(
            all_pred_labels, all_targets,
            task="multiclass",
            num_classes=self.cfg.num_classes,
            average=None,
        )
        for i, cls_name in enumerate(self.cfg.class_names):
            self.log(f"val/f1_{cls_name}", per_class_f1[i], on_epoch=True)

        # Accuracy
        acc = accuracy(all_pred_labels, all_targets, task="multiclass", num_classes=self.cfg.num_classes)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)

        # Confusion matrix (log as figure to TensorBoard)
        cm = confusion_matrix(all_pred_labels, all_targets, task="multiclass", num_classes=self.cfg.num_classes)
        if self.logger and hasattr(self.logger, "experiment"):
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm.numpy().astype(int),
                annot=True, fmt="d", cmap="Blues",
                xticklabels=self.cfg.class_names,
                yticklabels=self.cfg.class_names,
                ax=ax,
            )
            ax.set_xlabel("Predicho")
            ax.set_ylabel("Real")
            ax.set_title(f"Matriz de Confusión — Época {self.current_epoch}")
            self.logger.experiment.add_figure("val/confusion_matrix", fig, self.current_epoch)
            plt.close(fig)

        # Clear buffers
        self._val_preds.clear()
        self._val_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.max_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def on_train_epoch_start(self) -> None:
        """Handle backbone freezing/unfreezing schedule."""
        if self.cfg.freeze_backbone_epochs <= 0:
            return
        if self.current_epoch == 0:
            # Freeze backbone at start
            self._set_backbone_requires_grad(False)
            logger.info("Época %d: backbone congelado", self.current_epoch)
        elif self.current_epoch == self.cfg.freeze_backbone_epochs:
            # Unfreeze backbone
            self._set_backbone_requires_grad(True)
            logger.info("Época %d: backbone descongelado", self.current_epoch)

    def _set_backbone_requires_grad(self, requires_grad: bool) -> None:
        for param in self.model.backbone.parameters():
            param.requires_grad = requires_grad


# ------------------------------------------------------------------ #
# Callbacks
# ------------------------------------------------------------------ #


def build_callbacks(config: ClassificationConfig, output_dir: Path) -> list:
    """Build EarlyStopping and ModelCheckpoint callbacks."""
    early_stop = EarlyStopping(
        monitor="val/macro_f1",
        mode="max",
        patience=config.patience,
        verbose=True,
    )

    checkpoint = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="best-{epoch:02d}-{val/macro_f1:.3f}",
        monitor="val/macro_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    return [early_stop, checkpoint]


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train wound type classifier"
    )
    parser.add_argument(
        "--csv-prefix", type=Path, default=Path("data-clasificador"),
        help="Directory containing train.csv and val.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/classifier"),
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num-classes", type=int, default=None,
        help="Number of classes (default: auto-detect from CSV)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 1 epoch smoke test, skip full training",
    )
    parser.add_argument(
        "--no-freeze", action="store_true",
        help="Skip backbone freezing phase",
    )
    parser.add_argument(
        "--freeze-epochs", type=int, default=3,
        help="Number of epochs to freeze backbone (default: 3). Ignored if --no-freeze.",
    )
    parser.add_argument(
        "--no-mask", action="store_true",
        help="Use 3-channel input (no mask channel)",
    )
    parser.add_argument(
        "--accumulate", type=int, default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--domain-adapt", action="store_true", default=True,
        help="Enable aggressive RandomResizedCrop (scale 0.3-0.8) to simulate "
             "pipeline bbox crops (default: True). Use --no-domain-adapt to disable.",
    )
    parser.add_argument(
        "--no-domain-adapt", action="store_false", dest="domain_adapt",
        help="Disable domain-adapted augmentations (use standard scale 0.6-1.0)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.4,
        help="Dropout probability before classification head (default: 0.4)",
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.1,
        help="Label smoothing factor for CrossEntropyLoss (default: 0.1)",
    )
    return parser.parse_args()


def _resolve_num_classes(config: ClassificationConfig) -> int:
    """Auto-detect number of classes from train CSV labels."""
    train_csv = config.train_csv
    if not train_csv.exists():
        return config.num_classes
    df = pd.read_csv(train_csv)
    actual_labels = set(df["label"].unique())
    # Intersect with known class names
    known = [c for c in config.class_names if c in actual_labels]
    if len(known) < config.num_classes:
        logger.warning(
            "CSV has %d classes (%s), config says %d. Using %d.",
            len(known), known, config.num_classes, len(known),
        )
        config.class_names = known
        config.num_classes = len(known)
    return config.num_classes


def main() -> None:
    args = parse_args()

    # Build config from CLI args
    config = ClassificationConfig(
        train_csv=args.csv_prefix / "train.csv",
        val_csv=args.csv_prefix / "val.csv",
        test_csv=args.csv_prefix / "test.csv",
        output_dir=args.output_dir,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        accumulation_steps=args.accumulate,
        use_mask=not args.no_mask,
        freeze_backbone_epochs=0 if args.no_freeze else args.freeze_epochs,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
    )

    # Auto-detect num_classes from CSV
    num_classes = _resolve_num_classes(config)
    config.num_classes = num_classes
    logger.info("Entrenando con %d clases: %s", num_classes, config.class_names)

    # Ensure output dir exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    dm = WoundDataModule(config, domain_adapt=args.domain_adapt)
    dm.setup("fit")

    # Model
    model = WoundClassifierLightning(
        config=config,
        class_weights=dm.class_weights,
    )

    # Callbacks
    callbacks = build_callbacks(config, config.output_dir)
    if args.dry_run:
        # Remove ModelCheckpoint for dry-run (EarlyStopping alone is fine)
        callbacks = [cb for cb in callbacks if not isinstance(cb, ModelCheckpoint)]

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=str(config.output_dir / "logs"),
        name="wound_classifier",
    )

    # Trainer
    effective_epochs = 1 if args.dry_run else config.max_epochs
    trainer = pl.Trainer(
        max_epochs=effective_epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        callbacks=callbacks,
        logger=tb_logger,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulation_steps,
        log_every_n_steps=5,
        enable_progress_bar=True,
    )

    logger.info("Iniciando entrenamiento...")
    logger.info("  Dispositivo: %s", DEVICE)
    logger.info("  Épocas: %d", effective_epochs)
    logger.info("  Batch size: %d (accum %d → efectivo %d)",
                config.batch_size, config.accumulation_steps,
                config.batch_size * config.accumulation_steps)
    logger.info("  Learning rate: %.2e", config.learning_rate)
    logger.info("  MixUp α: %.1f  CutMix α: %.1f", config.mixup_alpha, config.cutmix_alpha)
    logger.info("  Label smoothing: %.2f  Dropout: %.2f  DomainAdapt: %s",
                config.label_smoothing, config.dropout, args.domain_adapt)
    logger.info("  Freeze backbone: %d epochs", config.freeze_backbone_epochs)
    logger.info("  Muestras train: %d  val: %d",
                len(dm.train_dataset), len(dm.val_dataset))  # type: ignore[arg-type]

    trainer.fit(model, datamodule=dm)

    # Final report
    if not args.dry_run:
        best_f1 = trainer.callback_metrics.get("val/macro_f1", 0)
        logger.info("Entrenamiento completado. Mejor macro-F1: %.4f", best_f1)
        best_path = trainer.checkpoint_callback.best_model_path  # type: ignore[union-attr]
        if best_path:
            logger.info("Mejor checkpoint: %s", best_path)
            # Copy best to a fixed name
            import shutil
            best_fixed = config.output_dir / "best.pth"
            shutil.copy(best_path, str(best_fixed))
            logger.info("Copiado a: %s", best_fixed)
    else:
        logger.info("Dry-run completado — verificar que loss no sea NaN")


if __name__ == "__main__":
    main()
