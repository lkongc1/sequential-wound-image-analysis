"""Segmentation metrics: Dice, IoU, Sensitivity, Specificity, Precision."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def calculate_metrics(preds, targets, threshold=0.5):
    """Calculate Dice, IoU, Sensitivity, Specificity, and Precision.

    Args:
        preds: Predicted logits or probabilities (any shape).
        targets: Ground-truth binary masks (same shape as preds).
        threshold: Binarization threshold when preds are probabilities.

    Returns:
        Dictionary with dice, iou, sensitivity, specificity, precision.
    """
    preds = (preds > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    sensitivity = (tp + 1e-6) / (tp + fn + 1e-6)
    specificity = (tn + 1e-6) / (tn + fp + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "sensitivity": sensitivity.item(),
        "specificity": specificity.item(),
        "precision": precision.item(),
    }


def _instance_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice coefficient between two binary instance masks."""
    intersection = float(np.logical_and(pred_mask, gt_mask).sum())
    total = float(pred_mask.sum()) + float(gt_mask.sum())
    if total < 1e-9:
        return 0.0
    return 2.0 * intersection / total


def _compute_iou_matrix(
    pred_ids: list[int],
    gt_ids: list[int],
    pred_map: np.ndarray,
    gt_map: np.ndarray,
) -> np.ndarray:
    """Build IoU matrix of shape (len(pred_ids), len(gt_ids))."""
    iou_mat = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float64)
    for i, pid in enumerate(pred_ids):
        p_mask = pred_map == pid
        for j, gid in enumerate(gt_ids):
            g_mask = gt_map == gid
            inter = float(np.logical_and(p_mask, g_mask).sum())
            union = float(np.logical_or(p_mask, g_mask).sum())
            iou_mat[i, j] = inter / union if union > 0 else 0.0
    return iou_mat


def per_instance_dice(
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
) -> dict[str, Any]:
    """Compute per-instance Dice between two uint16 instance maps.

    Matches predicted instances to ground-truth instances via IoU using
    Hungarian (Munkres) assignment. Each predicted instance is paired with
    the best-matching GT instance; unmatched predictions or GT instances
    contribute a Dice of 0.0 to the reported mean/median.

    Args:
        pred_instances: uint16 map, 0 = background, 1..N = instances.
        gt_instances: uint16 map, 0 = background, 1..N = instances.

    Returns:
        dict with keys:
            mean_dice: Mean Dice across all matched + unmatched instances.
            median_dice: Median Dice.
            per_instance: List of dicts with keys:
                pred_id, gt_id, iou, dice (one per matched pair).
            matched_pairs: Number of Hungarian-matched pairs.
    """
    pred_ids = sorted(set(pred_instances[pred_instances > 0]))
    gt_ids = sorted(set(gt_instances[gt_instances > 0]))

    # Edge case: both empty → perfect score
    if not pred_ids and not gt_ids:
        return {
            "mean_dice": 1.0,
            "median_dice": 1.0,
            "per_instance": [],
            "matched_pairs": 0,
        }

    # Edge case: one empty → 0.0
    if not pred_ids or not gt_ids:
        n = max(len(pred_ids), len(gt_ids))
        return {
            "mean_dice": 0.0,
            "median_dice": 0.0,
            "per_instance": [],
            "matched_pairs": 0,
        }

    # Build IoU cost matrix and run Hungarian assignment
    iou_mat = _compute_iou_matrix(pred_ids, gt_ids, pred_instances, gt_instances)
    # Hungarian minimises cost; use 1 - IoU as cost
    cost = 1.0 - iou_mat
    row_idx, col_idx = linear_sum_assignment(cost)

    per_instance: list[dict[str, Any]] = []
    dices: list[float] = []

    # Track which GT instances were matched
    matched_gt: set[int] = set()
    for r, c in zip(row_idx, col_idx):
        pid = pred_ids[r]
        gid = gt_ids[c]
        iou_val = float(iou_mat[r, c])
        dice_val = _instance_dice(pred_instances == pid, gt_instances == gid)
        per_instance.append({
            "pred_id": pid,
            "gt_id": gid,
            "iou": round(iou_val, 6),
            "dice": round(dice_val, 6),
        })
        dices.append(dice_val)
        matched_gt.add(gid)

    # Unmatched predictions → 0.0 Dice
    for pid in pred_ids:
        # Already matched?
        if any(entry["pred_id"] == pid for entry in per_instance):
            continue
        dices.append(0.0)
        per_instance.append({
            "pred_id": pid,
            "gt_id": 0,
            "iou": 0.0,
            "dice": 0.0,
        })

    # Unmatched GT instances → 0.0 Dice
    for gid in gt_ids:
        if gid in matched_gt:
            continue
        dices.append(0.0)
        per_instance.append({
            "pred_id": 0,
            "gt_id": gid,
            "iou": 0.0,
            "dice": 0.0,
        })

    dice_arr = np.array(dices)
    return {
        "mean_dice": round(float(dice_arr.mean()), 6),
        "median_dice": round(float(np.median(dice_arr)), 6),
        "per_instance": per_instance,
        "matched_pairs": len(per_instance),
    }
