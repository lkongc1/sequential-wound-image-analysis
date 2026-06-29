"""Confusion matrix visualization using matplotlib/seaborn.

Generates heatmaps with explicit TP/FP/FN/TN labels, absolute counts,
percentages, and key clinical metrics. Designed for FDA 510(k) evidence.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.metrics.confusion import ConfusionMatrix


class ConfusionMatrixVisualizer:
    """Generates confusion matrix heatmaps with explicit TP/TN/FP/FN labels.

    Methods:
        plot_absolute(cm, model_name, output_path): Absolute count heatmap
            with per-cell TP/FP/FN/TN labels + percentages.
        plot_normalized(cm, model_name, output_path): Normalized heatmap
            with per-cell proportions.
        plot_comparison(cms, output_path): Multi-model 2x2 grid showing
            all confusion matrices side by side with metrics summary.
    """

    # Cell labels matching the matrix layout [[TN, FP], [FN, TP]]
    _CELL_LABELS = np.array([
        ["VERDADERO\nNEGATIVO\n(TN)",  "FALSO\nPOSITIVO\n(FP)"],
        ["FALSO\nNEGATIVO\n(FN)",      "VERDADERO\nPOSITIVO\n(TP)"],
    ])

    def __init__(self, figsize: tuple[int, int] = (8, 6)):
        self.figsize = figsize

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #

    def plot_absolute(
        self,
        cm: ConfusionMatrix,
        model_name: str = "model",
        output_path: Path | None = None,
    ) -> plt.Figure:
        """Plot absolute-count confusion matrix with TP/TN/FP/FN labels.

        Each cell shows:
          - Type label (TP / TN / FP / FN)
          - Absolute count (e.g. "27.1M")
          - Percentage of total pixels (e.g. "3.3%")
        """
        abs_mat, _ = cm.to_numpy()
        metrics = cm.derive_metrics()
        total = abs_mat.sum()

        fig, ax = plt.subplots(figsize=(9, 7))

        # Custom colormap: green for correct (TN, TP), red for errors (FP, FN)
        custom_cmap = sns.diverging_palette(130, 10, s=80, l=55, as_cmap=True)

        # Build annotation matrix with both counts and percentages
        annot = np.empty((2, 2), dtype=object)
        pct_cells = np.zeros((2, 2))

        labels_flat = [
            ("VERDADERO\nNEGATIVO\n(TN)",  cm.tn),
            ("FALSO\nPOSITIVO\n(FP)",      cm.fp),
            ("FALSO\nNEGATIVO\n(FN)",      cm.fn),
            ("VERDADERO\nPOSITIVO\n(TP)",  cm.tp),
        ]
        for idx, (label, count) in enumerate(labels_flat):
            r, c = divmod(idx, 2)
            pct = count / total * 100 if total > 0 else 0
            pct_cells[r][c] = pct
            count_str = self._fmt_count(count)
            annot[r][c] = f"{label}\n{count_str}\n({pct:.1f}%)"

        # Heatmap
        sns.heatmap(
            abs_mat, annot=annot, fmt="",
            cmap=custom_cmap, center=abs_mat.mean() if abs_mat.max() > 0 else 0,
            xticklabels=["PRED: NEGATIVO", "PRED: POSITIVO"],
            yticklabels=["REAL: NEGATIVO", "REAL: POSITIVO"],
            ax=ax, cbar_kws={"label": "Pixeles"},
            linewidths=2, linecolor="white",
        )

        # Title with key metrics
        ax.set_title(
            f"Matriz de Confusion — {model_name}\n"
            f"Dice={metrics['dice']:.3f} | "
            f"Sens={metrics['sensitivity']:.3f} | "
            f"Spec={metrics['specificity']:.3f} | "
            f"F2={metrics['f2']:.3f}",
            fontsize=12, fontweight="bold", pad=20,
        )

        ax.set_xlabel("PREDICCION DEL MODELO", fontsize=10, fontweight="bold")
        ax.set_ylabel("REALIDAD (Ground Truth)", fontsize=10, fontweight="bold")

        # Color-coded borders: green = correct, red = error
        for (r, c), color in {
            (0, 0): "#27ae60",  # TN — green
            (0, 1): "#e74c3c",  # FP — red
            (1, 0): "#e74c3c",  # FN — red
            (1, 1): "#27ae60",  # TP — green
        }.items():
            ax.add_patch(plt.Rectangle(
                (c, r), 1, 1, fill=False,
                edgecolor=color, linewidth=3,
            ))

        fig.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight",
                        facecolor="white")
            plt.close(fig)

        return fig

    def plot_normalized(
        self,
        cm: ConfusionMatrix,
        model_name: str = "model",
        output_path: Path | None = None,
    ) -> plt.Figure:
        """Plot normalized confusion matrix (proportions of total pixels)."""
        _, norm_mat = cm.to_numpy()
        abs_mat, _ = cm.to_numpy()
        total = abs_mat.sum()
        metrics = cm.derive_metrics()

        fig, ax = plt.subplots(figsize=(9, 7))

        # Annotations: percentage with 1 decimal
        annot = np.empty((2, 2), dtype=object)
        labels_flat = [
            ("TN", cm.tn), ("FP", cm.fp),
            ("FN", cm.fn), ("TP", cm.tp),
        ]
        for idx, (label, count) in enumerate(labels_flat):
            r, c = divmod(idx, 2)
            pct = count / total * 100 if total > 0 else 0
            annot[r][c] = f"{label}\n{pct:.2f}%"

        sns.heatmap(
            norm_mat, annot=annot, fmt="",
            cmap="YlOrRd",
            xticklabels=["PRED: NEGATIVO", "PRED: POSITIVO"],
            yticklabels=["REAL: NEGATIVO", "REAL: POSITIVO"],
            ax=ax, vmin=0, vmax=1,
            cbar_kws={"label": "Proporcion"},
            linewidths=2, linecolor="white",
        )

        ax.set_title(
            f"Matriz Normalizada — {model_name}\n"
            f"(Proporcion del total de {self._fmt_count(total)} pixeles)",
            fontsize=12, fontweight="bold", pad=20,
        )
        ax.set_xlabel("PREDICCION DEL MODELO", fontsize=10, fontweight="bold")
        ax.set_ylabel("REALIDAD (Ground Truth)", fontsize=10, fontweight="bold")

        fig.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight",
                        facecolor="white")
            plt.close(fig)

        return fig

    def plot_comparison(
        self,
        cms: dict[str, ConfusionMatrix],
        output_path: Path | None = None,
    ) -> plt.Figure:
        """Plot a 2×N grid comparing absolute confusion matrices across models.

        Each subplot shows the full annotated confusion matrix for one model.
        """
        n = len(cms)
        if n == 0:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No hay modelos para comparar",
                    ha="center", va="center", fontsize=14)
            ax.axis("off")
            if output_path:
                fig.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
            return fig

        cols = min(n, 2)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * 10, rows * 8),
            squeeze=False,
        )

        axes_flat = axes.flatten()

        for i, (name, cm) in enumerate(cms.items()):
            ax = axes_flat[i]
            abs_mat, _ = cm.to_numpy()
            metrics = cm.derive_metrics()
            total = abs_mat.sum()

            # Build annotations
            annot = np.empty((2, 2), dtype=object)
            labels_flat_local = [
                ("VERDADERO\nNEGATIVO (TN)", cm.tn),
                ("FALSO\nPOSITIVO (FP)",     cm.fp),
                ("FALSO\nNEGATIVO (FN)",     cm.fn),
                ("VERDADERO\nPOSITIVO (TP)", cm.tp),
            ]
            for idx, (label, count) in enumerate(labels_flat_local):
                r, c = divmod(idx, 2)
                pct = count / total * 100 if total > 0 else 0
                count_str = self._fmt_count(count)
                annot[r][c] = f"{label}\n{count_str}\n({pct:.1f}%)"

            custom_cmap = sns.diverging_palette(130, 10, s=80, l=55, as_cmap=True)

            sns.heatmap(
                abs_mat, annot=annot, fmt="",
                cmap=custom_cmap,
                center=abs_mat.mean() if abs_mat.max() > 0 else 0,
                xticklabels=["NEGATIVO", "POSITIVO"],
                yticklabels=["REAL NEG", "REAL POS"],
                ax=ax, cbar_kws={"label": "Pixeles"},
                linewidths=2, linecolor="white",
            )

            ax.set_title(
                f"{name}\n"
                f"Dice={metrics['dice']:.3f}  "
                f"Sens={metrics['sensitivity']:.3f}  "
                f"Spec={metrics['specificity']:.3f}  "
                f"F2={metrics['f2']:.3f}",
                fontsize=11, fontweight="bold", pad=15,
            )
            ax.set_xlabel("PREDICCION", fontsize=9)
            ax.set_ylabel("REALIDAD", fontsize=9)

            # Color borders
            for (r, c), color in {
                (0, 0): "#27ae60", (0, 1): "#e74c3c",
                (1, 0): "#e74c3c", (1, 1): "#27ae60",
            }.items():
                ax.add_patch(plt.Rectangle(
                    (c, r), 1, 1, fill=False,
                    edgecolor=color, linewidth=3,
                ))

        # Hide unused subplots
        for i in range(n, len(axes_flat)):
            axes_flat[i].set_visible(False)

        fig.suptitle(
            "COMPARATIVA DE MATRICES DE CONFUSION\n"
            "Verde = Acierto  |  Rojo = Error",
            fontsize=16, fontweight="bold", y=1.01,
        )
        fig.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight",
                        facecolor="white")
            plt.close(fig)

        return fig

    # ---------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------- #

    @staticmethod
    def _fmt_count(n: int) -> str:
        """Format large integers for readability: 27.1M, 754.9K, etc."""
        if abs(n) >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if abs(n) >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)
