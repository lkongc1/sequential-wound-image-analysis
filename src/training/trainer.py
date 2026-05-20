"""Training pipeline: Trainer with dependency injection, mixed precision.

This module provides the main :class:`Trainer` for wound-segmentation
models.  It uses automatic mixed precision (AMP), gradient scaling,
checkpointing, and early stopping.  All collaborators (loss, optimizer,
scheduler, callbacks) are injected via the constructor.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from src.training.checkpoint_manager import CheckpointManager
from src.training.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for segmentation models with dependency injection.

    Parameters
    ----------
    model:
        PyTorch model to train.
    criterion:
        Loss function (e.g. ``BCEDiceLoss()``).
    optimizer_factory:
        Callable that receives ``model.parameters()`` and returns an
        :class:`torch.optim.Optimizer`.
    scheduler_factory:
        Optional callable that receives the optimizer and returns a
        learning-rate scheduler.
    early_stopping:
        Optional :class:`EarlyStopping` instance.
    checkpoint_manager:
        Optional :class:`CheckpointManager` instance.
    device:
        ``torch.device`` or device-string (``"cuda"``, ``"cpu"``, …).
    config:
        Optional dict with hyper-parameters and paths (kept for backward
        compatibility).
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer_factory: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer],
        scheduler_factory: Callable[[torch.optim.Optimizer], Any] | None = None,
        early_stopping: EarlyStopping | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        device: torch.device | str | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.model = model
        self.criterion = criterion
        self.device = (
            torch.device(device)
            if isinstance(device, str)
            else (device or torch.device("cpu"))
        )
        self.optimizer = optimizer_factory(model.parameters())
        self.scheduler = (
            scheduler_factory(self.optimizer) if scheduler_factory else None
        )
        self.early_stopping = early_stopping
        self.checkpoint_manager = checkpoint_manager
        self.config = config or {}
        self.history: list[dict[str, float]] = []
        self.scaler = GradScaler(device=self.device.type)
        self._best_val_dice = -1.0

        # Ensure model is on the right device
        self.model.to(self.device)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
    ) -> list[dict[str, float]]:
        """Run the full training loop.

        Returns the training history (one entry per epoch).
        """
        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # ---- training ----
            train_loss = self.train_epoch(train_loader)

            # ---- validation ----
            val_metrics = self.validate(val_loader)
            val_dice = val_metrics["dice"]
            val_loss = val_metrics["loss"]

            # ---- scheduling ----
            if self.scheduler is not None:
                self.scheduler.step(val_dice)

            # ---- history ----
            epoch_info = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_metrics["iou"],
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(epoch_info)

            elapsed = time.time() - start_time
            logger.info(
                "Epoch %d/%d | train_loss: %.4f | val_loss: %.4f | "
                "val_dice: %.4f | val_iou: %.4f | lr: %.6f | %.1fs",
                epoch,
                epochs,
                train_loss,
                val_loss,
                val_dice,
                val_metrics["iou"],
                self.optimizer.param_groups[0]["lr"],
                elapsed,
            )

            # ---- checkpointing ----
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=epoch_info,
                )
            else:
                # fallback inline checkpointing for backward compatibility
                if val_dice > self._best_val_dice:
                    self._best_val_dice = val_dice
                    ckpt_dir = Path(
                        self.config.get("checkpoint_dir", "models/checkpoints/trainer")
                    )
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = ckpt_dir / "best.pth"
                    self.save_checkpoint(epoch=epoch, metrics=epoch_info, path=ckpt_path)
                    logger.info("New best model saved to %s", ckpt_path)

            # ---- early stopping ----
            if self.early_stopping is not None:
                self.early_stopping(val_dice)
                if self.early_stopping.early_stop:
                    logger.info("Stopping early at epoch %d", epoch)
                    break

            # ---- memory logging (optional) ----
            if self.device.type == "cuda":
                mem_alloc = torch.cuda.memory_allocated(self.device) / 1e9
                mem_reserved = torch.cuda.memory_reserved(self.device) / 1e9
                logger.info(
                    "GPU memory -- allocated: %.2f GB | reserved: %.2f GB",
                    mem_alloc,
                    mem_reserved,
                )

        return self.history

    def train_epoch(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> float:
        """Run a single training epoch with AMP."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> dict[str, float]:
        """Evaluate on a validation set.

        Returns a dict with ``loss``, ``dice``, and ``iou``.
        """
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = 0

        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            total_loss += loss.item()
            total_dice += _dice_score(outputs, masks)
            total_iou += _iou_score(outputs, masks)
            num_batches += 1

        denom = max(num_batches, 1)
        return {
            "loss": total_loss / denom,
            "dice": total_dice / denom,
            "iou": total_iou / denom,
        }

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, Any],
        path: Path | str | None = None,
    ) -> None:
        """Persist model weights and metadata to disk."""
        if path is None:
            ckpt_dir = Path(
                self.config.get("checkpoint_dir", "models/checkpoints/trainer")
            )
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = ckpt_dir / f"epoch_{epoch:03d}.pth"
        else:
            path = Path(path)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "history": self.history,
        }
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: Path | str) -> None:
        """Restore model weights from a checkpoint file."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", [])
        self._best_val_dice = max(
            (entry.get("val_dice", -1.0) for entry in self.history),
            default=-1.0,
        )
        logger.info("Checkpoint loaded from %s", path)

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
    ) -> dict[str, float]:
        """Final evaluation on a held-out test set.

        Returns a dict with ``loss``, ``dice``, and ``iou``.
        """
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = 0

        for images, masks in test_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            total_loss += loss.item()
            total_dice += _dice_score(outputs, masks)
            total_iou += _iou_score(outputs, masks)
            num_batches += 1

        denom = max(num_batches, 1)
        return {
            "loss": total_loss / denom,
            "dice": total_dice / denom,
            "iou": total_iou / denom,
        }


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _dice_score(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
    """Compute Dice coefficient from raw logits."""
    probs = torch.sigmoid(logits)
    probs = probs.view(-1)
    targets = targets.view(-1)
    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def _iou_score(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
    """Compute IoU (Jaccard index) from raw logits."""
    probs = torch.sigmoid(logits)
    probs = probs.view(-1)
    targets = targets.view(-1)
    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()
