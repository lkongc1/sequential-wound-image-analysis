"""Checkpoint manager for saving and loading training checkpoints."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Handle saving/loading of model checkpoints with metric-based filtering.

    Parameters
    ----------
    output_dir:
        Directory where checkpoints will be persisted.
    monitor:
        Metric name to track for deciding the *best* checkpoint.
    mode:
        ``"max"`` if higher *monitor* is better, ``"min"`` otherwise.
    """

    def __init__(
        self,
        output_dir: Path,
        monitor: str = "val_dice",
        mode: str = "max",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self._best_score: float | None = None
        self._best_path: Path | None = None

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, Any],
    ) -> Path | None:
        """Save a checkpoint if *metrics* improved over the best seen so far.

        Returns the path of the saved checkpoint, or ``None`` if the checkpoint
        was not better than the current best.
        """
        current = metrics.get(self.monitor)
        if current is None:
            logger.warning("Monitor metric '%s' not found in metrics", self.monitor)
            return None

        if not self.is_better(current, self._best_score):
            return None

        self._best_score = current
        path = self.output_dir / "best.pth"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, path)
        logger.info("New best checkpoint saved to %s (%s=%.4f)", path, self.monitor, current)
        self._best_path = path
        return path

    def load(self, path: Path) -> dict[str, Any]:
        """Load a checkpoint from disk.

        Returns a dict containing ``model_state_dict``, ``optimizer_state_dict``,
        ``epoch``, and ``metrics``.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        logger.info("Checkpoint loaded from %s", path)
        return checkpoint

    def is_better(self, current: float, best: float | None) -> bool:
        """Return ``True`` if *current* is better than *best* according to *mode*."""
        if best is None:
            return True
        if self.mode == "max":
            return current > best
        return current < best
