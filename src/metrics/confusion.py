"""Confusion matrix accumulation and metric derivation for wound segmentation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ConfusionMatrix:
    """Accumulates pixel-level TP/FP/FN/TN across batches and derives clinical metrics.

    Usage:
        cm = ConfusionMatrix()
        cm.accumulate(preds, targets)
        metrics = cm.derive_metrics()
        abs_mat, norm_mat = cm.to_numpy()
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def accumulate(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
    ) -> None:
        """Binarize predictions, compute pixel-level TP/FP/FN/TN, add to running totals.

        Args:
            preds: Predicted probabilities of shape (B, C, H, W) or (B, H, W).
            targets: Ground-truth binary masks, same shape as preds.
            threshold: Binarization threshold (default 0.5).
        """
        preds_bin = (preds > threshold).float()
        targets = targets.float()

        tp_count = int((preds_bin * targets).sum().item())
        fp_count = int((preds_bin * (1 - targets)).sum().item())
        fn_count = int(((1 - preds_bin) * targets).sum().item())
        tn_count = int(((1 - preds_bin) * (1 - targets)).sum().item())

        self.tp += tp_count
        self.fp += fp_count
        self.fn += fn_count
        self.tn += tn_count

    def derive_metrics(self) -> dict[str, float]:
        """Derive clinical and segmentation metrics from accumulated counts.

        Returns:
            Dictionary with keys: sensitivity, specificity, precision, npv,
            f2, dice, iou, accuracy. All values are 0.0 when no predictions exist.
        """
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # F2-score: beta=2 weights recall (sensitivity) twice as important as precision
        beta = 2.0
        beta2 = beta * beta
        f2_denom = (beta2 * tp) + (beta2 * fn) + fp
        f2 = ((1 + beta2) * tp) / f2_denom if f2_denom > 0 else 0.0

        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

        return {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "npv": npv,
            "f2": f2,
            "dice": dice,
            "iou": iou,
            "accuracy": accuracy,
        }

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return confusion matrix as numpy arrays for heatmap plotting.

        Returns:
            Tuple of (absolute_2x2, normalized_2x2).
            Matrix layout: rows = actual, cols = predicted:
                [[TN, FP], [FN, TP]]
        """
        absolute = np.array(
            [[self.tn, self.fp], [self.fn, self.tp]],
            dtype=np.int64,
        )

        total = absolute.sum()
        if total > 0:
            normalized = absolute.astype(np.float64) / total
        else:
            normalized = np.zeros((2, 2), dtype=np.float64)

        return absolute, normalized
