"""Visualization generators following SOLID principles (SRP).

SRP: Only handles visualizations, nothing else.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class WoundDistributionVisualizer:
    """Generates EDA visualizations for wound segmentation data.

    Single Responsibility: Only generates and saves plots.

    Example:
        >>> viz = WoundDistributionVisualizer(output_dir=Path("output"))
        >>> viz.visualize(df)
    """

    def __init__(self, output_dir: Path):
        """Initialize visualizer.

        Args:
            output_dir: Directory to save generated plots.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize(self, df: pd.DataFrame, skip_plots: bool = False) -> None:
        """Generate all EDA visualizations and save to output_dir.

        Args:
            df: DataFrame with per-file statistics.
            skip_plots: If True, skip visualization generation.
        """
        if skip_plots:
            return

        self._plot_wound_analysis(df)
        logger.info(f"Saved figures to {self.output_dir}")

    def _plot_wound_analysis(self, df: pd.DataFrame) -> None:
        """Generate comprehensive wound analysis plots.

        Creates a 2x2 grid with:
        - Histogram of wound percentage
        - Boxplot by source
        - Violin plot of brightness by source
        - Scatter plot brightness vs wound area
        """
        fig1, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Histogram of wound %
        wound_pct = df["mask_wound_percentage"].dropna()
        ax = axes[0, 0]
        if len(wound_pct) > 0:
            sns.histplot(data=None, x=wound_pct.to_numpy(), kde=True, bins=50, color="steelblue", ax=ax)
            ax.axvline(wound_pct.mean(), color="red", linestyle="--", label=f"Mean: {wound_pct.mean():.2f}%")
            ax.axvline(wound_pct.median(), color="orange", linestyle="--", label=f"Median: {wound_pct.median():.2f}%")
            ax.set_title("Wound Area Distribution", fontweight="bold")
            ax.set_xlabel("Wound Area (%)")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(alpha=0.3)

        # Plot 2: Boxplot by source
        ax = axes[0, 1]
        order = df.groupby("source")["mask_wound_percentage"].median().sort_values().index
        sns.boxplot(data=df, x="source", y="mask_wound_percentage", order=order, palette="Set2", ax=ax, hue="source", legend=False)
        ax.set_title("Distribution by Source", fontweight="bold")
        ax.set_xlabel("Source")
        ax.set_ylabel("Wound Area (%)")
        ax.tick_params(axis="x", rotation=45)

        # Plot 3: Violin plot of brightness by source
        ax = axes[1, 0]
        sns.violinplot(data=df, x="source", y="image_brightness_mean", order=order, palette="Set2", ax=ax, hue="source", legend=False)
        ax.set_title("Mean Brightness by Source", fontweight="bold")
        ax.set_xlabel("Source")
        ax.set_ylabel("Brightness (0-255)")
        ax.tick_params(axis="x", rotation=45)

        # Plot 4: Scatter brightness vs wound %
        ax = axes[1, 1]
        colors = {"fusc": "#1f77b4", "wsnet": "#2ca02c", "medetec": "#d62728", "unknown": "gray"}
        for source in df["source"].unique():
            src_df = df[df["source"] == source]
            if len(src_df) > 0:
                ax.scatter(
                    src_df["image_brightness_mean"].to_numpy(),
                    src_df["mask_wound_percentage"].to_numpy(),
                    alpha=0.4, label=source, c=colors.get(source, "gray"), s=15,
                )
        ax.set_title("Brightness vs Wound Area", fontweight="bold")
        ax.set_xlabel("Mean Brightness")
        ax.set_ylabel("Wound Area (%)")
        ax.legend(fontsize=8, loc="best")

        plt.tight_layout()
        fig_path = self.output_dir / "fig1_wound_analysis.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig1)
