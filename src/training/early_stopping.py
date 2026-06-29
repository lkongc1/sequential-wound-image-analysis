"""Early stopping callback for training loops."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Track a validation metric and signal when training should stop.

    Parameters
    ----------
    patience:
        How many epochs to wait without improvement before stopping.
    min_delta:
        Minimum change to count as an improvement.
    mode:
        ``"max"`` for metrics that should be maximised (e.g. Dice) or
        ``"min"`` for metrics that should be minimised (e.g. loss).
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False

    def __call__(self, score: float) -> None:
        """Update state with the latest validation score."""
        if self.best_score is None:
            self.best_score = score
            return

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    "Early stopping triggered after %d epochs without improvement. "
                    "Best score: %.4f",
                    self.patience,
                    self.best_score,
                )
