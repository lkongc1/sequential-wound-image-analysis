"""Comparative reporter: generates Markdown comparison tables and FDA 510(k)
clinical justification documents.

Follows the ``BaseReporter`` / ``CSVReporter`` pattern from
``src/data/eda/reporters.py``.
"""
from __future__ import annotations

from typing import Any

# FDA 510(k) clinical thresholds
CLINICAL_THRESHOLDS = {
    "sensitivity": 0.90,
    "specificity": 0.95,
    "dice": 0.85,
    "npv": 0.95,
}


class ComparativeReporter:
    """Generates markdown-based reports for model comparison."""

    METRIC_COLUMNS = [
        "Sensitivity",
        "Specificity",
        "Precision",
        "NPV",
        "F2",
        "Dice",
        "IoU",
        "Accuracy",
    ]

    METRIC_KEYS = [
        "sensitivity",
        "specificity",
        "precision",
        "npv",
        "f2",
        "dice",
        "iou",
        "accuracy",
    ]

    def generate_markdown(self, results: dict[str, dict[str, Any]]) -> str:
        """Generate a Markdown comparison table from evaluation results.

        Parameters
        ----------
        results:
            Dict mapping model_name → {"confusion": ConfusionMatrix, "metrics": dict}.

        Returns
        -------
        Markdown string with header, table, and threshold notes.
        """
        if not results:
            return (
                "# Comparative Evaluation Report\n\n"
                "**No models were evaluated.** Verify that checkpoints exist for "
                "the configured model names.\n"
            )

        lines = [
            "# Comparative Evaluation Report",
            "",
            "## Model Comparison",
            "",
        ]

        # Header row
        header = "| Model | " + " | ".join(self.METRIC_COLUMNS) + " |"
        lines.append(header)

        # Separator
        sep = "|-------|" + "|".join(["-------" for _ in self.METRIC_COLUMNS]) + "|"
        lines.append(sep)

        # Data rows
        for name in sorted(results.keys()):
            metrics = results[name].get("metrics", {})
            vals = []
            for key in self.METRIC_KEYS:
                val = metrics.get(key, 0.0)
                vals.append(f"{val:.4f}")
            row = f"| {name} | " + " | ".join(vals) + " |"
            lines.append(row)

        lines.extend([
            "",
            "## Clinical Thresholds (FDA 510(k))",
            "",
            "| Metric      | Threshold |",
            "|-------------|-----------|",
            f"| Sensitivity | ≥ {CLINICAL_THRESHOLDS['sensitivity']:.2f}     |",
            f"| Specificity | ≥ {CLINICAL_THRESHOLDS['specificity']:.2f}     |",
            f"| Dice        | ≥ {CLINICAL_THRESHOLDS['dice']:.2f}     |",
            f"| NPV         | ≥ {CLINICAL_THRESHOLDS['npv']:.2f}     |",
            "",
        ])

        return "\n".join(lines) + "\n"

    def generate_clinical_justification(self, results: dict[str, dict[str, Any]]) -> str:
        """Generate FDA 510(k) clinical justification document.

        Parameters
        ----------
        results:
            Dict mapping model_name → {"confusion": ..., "metrics": dict}.

        Returns
        -------
        Markdown string with Methodology, Results, Clinical Reasoning, Recommendation.
        """
        if not results:
            return (
                "# Clinical Justification\n\n"
                "**No models were evaluated.** Cannot produce clinical justification "
                "without evaluation results.\n"
            )

        # Find best model by F2-score (FN-weighted)
        ranked = sorted(
            results.items(),
            key=lambda item: item[1].get("metrics", {}).get("f2", 0.0),
            reverse=True,
        )
        best_name, best_data = ranked[0]
        best_metrics = best_data.get("metrics", {})

        # Build threshold compliance table for the best model
        threshold_rows = []
        for metric_name, threshold in CLINICAL_THRESHOLDS.items():
            value = best_metrics.get(metric_name, 0.0)
            passed = "PASS" if value >= threshold else "FAIL"
            threshold_rows.append(f"| {metric_name.capitalize()} | ≥ {threshold:.2f} | {value:.4f} | {passed} |")

        lines = [
            "# Clinical Justification for FDA 510(k) Submission",
            "",
            "## Methodology",
            "",
            "Four wound segmentation models (U-Net, Attention U-Net, U-Net++, DeepLabV3+) were evaluated "
            "on a held-out test set using patient-aware splitting to prevent data leakage. Pixel-level "
            "confusion matrices were accumulated across all test images and clinical metrics derived "
            "globally (micro-averaged).",
            "",
            "Evaluation metrics include: sensitivity, specificity, precision, NPV, F2-score, Dice, IoU, "
            "and accuracy. F2-score (beta=2) prioritizes recall (sensitivity) to minimize missed wound "
            "detections — the primary clinical safety concern.",
            "",
            "## Results",
            "",
            "### Model Rankings (by F2-score)",
            "",
            "| Rank | Model | F2 | Dice | Sensitivity | Specificity |",
            "|------|-------|----|------|-------------|-------------|",
        ]

        for rank, (name, data) in enumerate(ranked, 1):
            m = data.get("metrics", {})
            lines.append(
                f"| {rank} | {name} | {m.get('f2', 0):.4f} | {m.get('dice', 0):.4f} | "
                f"{m.get('sensitivity', 0):.4f} | {m.get('specificity', 0):.4f} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "*Generated by ComparativeEvaluator -- Wound Segmentation Project*",
            "",
        ])
        lines.extend(threshold_rows)

        # Clinical Reasoning
        lines.extend([
            "",
            "## Clinical Reasoning",
            "",
            f"The **{best_name}** model was selected as the recommended model for FDA 510(k) "
            "submission based on the following clinical criteria:",
            "",
        ])

        # List of clinical justifications
        f2 = best_metrics.get("f2", 0)
        sensitivity = best_metrics.get("sensitivity", 0)
        specificity = best_metrics.get("specificity", 0)
        dice = best_metrics.get("dice", 0)
        npv_val = best_metrics.get("npv", 0)

        lines.append(f"- **Recall-weighted performance**: F2-score of {f2:.4f} demonstrates strong "
                       "ability to detect wound pixels while minimizing false negatives — the most "
                       "critical clinical concern (missed wounds).")

        if sensitivity >= CLINICAL_THRESHOLDS["sensitivity"]:
            lines.append(f"- **Sensitivity ≥ 0.90**: Achieved {sensitivity:.4f}, meeting the clinical "
                           "threshold for wound detection sensitivity.")
        else:
            lines.append(f"- **Sensitivity**: At {sensitivity:.4f}, below the 0.90 threshold. "
                           "Further training or data augmentation recommended.")

        if specificity >= CLINICAL_THRESHOLDS["specificity"]:
            lines.append(f"- **Specificity ≥ 0.95**: Achieved {specificity:.4f}, ensuring low false "
                           "positive rate and preventing unnecessary clinical interventions.")
        else:
            lines.append(f"- **Specificity**: At {specificity:.4f}, below the 0.95 threshold.")

        if dice >= CLINICAL_THRESHOLDS["dice"]:
            lines.append(f"- **Dice ≥ 0.85**: Achieved {dice:.4f}, confirming strong spatial overlap "
                           "between predicted and ground-truth segmentations.")
        else:
            lines.append(f"- **Dice**: At {dice:.4f}, below the 0.85 threshold.")

        if npv_val >= CLINICAL_THRESHOLDS["npv"]:
            lines.append(f"- **NPV ≥ 0.95**: Achieved {npv_val:.4f}, indicating high confidence that "
                           "pixels classified as non-wound are truly non-wound.")

        lines.extend([
            "",
            "## Recommendation",
            "",
            f"**Model recommended for FDA 510(k) submission:** `{best_name}`",
            "",
            f"The {best_name} model ranked highest by F2-score (recall-weighted), which prioritizes "
            "minimizing false negatives — missed wound detections are the primary clinical safety risk. "
            "The model's sensitivity, specificity, and Dice scores were evaluated against FDA clinical "
            "thresholds to ensure suitability for regulatory submission.",
            "",
            "### Caveats",
            "",
            "- Results assume the test set is representative of the target clinical population.",
            "- External clinical validation with prospective data is recommended before final submission.",
            "- Model performance should be stratified by wound type, size, and anatomical location "
            "for comprehensive regulatory evidence.",
            "",
            "---",
            "",
            "*Generated by ComparativeEvaluator — Wound Segmentation Project*",
            "",
        ])

        return "\n".join(lines) + "\n"
