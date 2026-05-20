"""Segmentation metrics: Dice, IoU, Sensitivity, Specificity, Precision."""


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
