"""Comparative model evaluator: orchestrates multi-model inference, confusion
accumulation, and reporting.

Follows SOLID: SRP — eval orchestrator; OCP — accepts config, model_factory,
reporter, visualizer via DI.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import ComparativeConfig
from src.inference.predictor import Predictor
from src.metrics.confusion import ConfusionMatrix
from src.models.factory import create_model

logger = logging.getLogger(__name__)


class ComparativeEvaluator:
    """Iterates over registered models, runs inference, accumulates confusion
    matrices, and delegates reporting/visualization.

    Parameters
    ----------
    config:
        ComparativeConfig instance controlling models, paths, and thresholds.
    reporter:
        Optional reporter instance (default: ComparativeReporter).
        If ``None``, reporting is skipped.
    visualizer:
        Optional visualizer instance (default: ConfusionMatrixVisualizer).
        If ``None``, heatmap generation is skipped.
    """

    def __init__(
        self,
        config: ComparativeConfig,
        reporter: Any = None,
        visualizer: Any = None,
    ):
        self.config = config
        self._reporter = reporter
        self._visualizer = visualizer

    @staticmethod
    def _find_checkpoint(checkpoint_dir: Path, model_name: str) -> Path | None:
        """Locate a checkpoint for *model_name* inside *checkpoint_dir*.

        Search order per candidate directory (name, trainer, name_*):
          1. ``{dir}/best.pth``
          2. ``{dir}/*.pth`` (latest by mtime)
          3. ``{dir}/best.ckpt``
          4. ``{dir}/*.ckpt`` (latest by mtime)

        Candidate directories tried in order:
          a. ``{checkpoint_dir}/{model_name}/``
          b. ``{checkpoint_dir}/trainer/`` (generic trainer checkpoint)
          c. ``{checkpoint_dir}/{model_name}_*/`` (e.g. unet_resnet50_mvp)

        Returns the ``Path`` of the first found checkpoint or ``None``.
        """
        base = Path(checkpoint_dir)

        # Build candidate directory list
        candidates: list[Path] = []

        # a. Exact model name dir
        exact = base / model_name
        if exact.is_dir():
            candidates.append(exact)

        # b. Trainer dir — only relevant for unet (the only model with Trainer checkpoints)
        if model_name == "unet":
            trainer_dir = base / "trainer"
            if trainer_dir.is_dir():
                candidates.append(trainer_dir)

        # c. Subdirs starting with model_name (e.g. unet_resnet50_mvp)
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and entry.name.startswith(f"{model_name}_"):
                candidates.append(entry)

        for cand in candidates:
            # 1. best.pth
            best_pth = cand / "best.pth"
            if best_pth.exists():
                return best_pth

            # 2. any .pth (latest)
            pth_files = sorted(
                cand.glob("*.pth"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if pth_files:
                return pth_files[0]

            # 3. best.ckpt
            best_ckpt = cand / "best.ckpt"
            if best_ckpt.exists():
                return best_ckpt

            # 4. any .ckpt (latest)
            ckpt_files = sorted(
                cand.glob("*.ckpt"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if ckpt_files:
                return ckpt_files[0]

        return None

    @staticmethod
    def _load_checkpoint(model: nn.Module, checkpoint_path: Path, device: str = "cpu") -> None:
        """Load checkpoint weights into *model*.

        Handles:
        - ``.pth`` with key ``model_state_dict`` (CheckpointManager format).
        - ``.ckpt`` with key ``state_dict`` and ``model.`` prefix stripping.
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if checkpoint_path.suffix == ".ckpt":
            # Lightning checkpoint → strip prefix
            state_dict = ckpt.get("state_dict", {})
            stripped = {
                k.replace("model.", "", 1) if k.startswith("model.") else k: v
                for k, v in state_dict.items()
            }
            model.load_state_dict(stripped, strict=False)
        else:
            # .pth (CheckpointManager format)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)

    def evaluate(
        self,
        data_loader: DataLoader,
        device: str | None = None,
    ) -> dict[str, dict[str, object]]:
        """Run comparative evaluation across all configured models.

        Parameters
        ----------
        data_loader:
            DataLoader providing (images, masks) batches. Masks must be
            binary (0/1) with shape ``(B, 1, H, W)``.
        device:
            Device string ("cuda", "cpu", etc.). Auto-detected if ``None``.

        Returns
        -------
        Dict mapping model_name → {
            "confusion": ConfusionMatrix,
            "metrics": dict[str, float],
            "checkpoint_path": str | None,
        }
        Models without checkpoints are absent from the result.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        results: dict[str, dict[str, object]] = {}

        for name in self.config.model_names:
            # Find checkpoint
            ckpt_path = self._find_checkpoint(self.config.checkpoint_base_dir, name)
            if ckpt_path is None:
                logger.warning(
                    "Skipping %s: no checkpoint found in %s/%s",
                    name, self.config.checkpoint_base_dir, name,
                )
                continue

            logger.info("Evaluating %s (checkpoint: %s)", name, ckpt_path)

            # Create model — SMP models accept pretrained, custom models don't.
            # Try with pretrained=False first (avoids downloading weights), 
            # fall back to bare create_model(name) for custom architectures.
            try:
                model = create_model(name, pretrained=False)
            except TypeError:
                model = create_model(name)
            except (ValueError, TypeError) as exc:
                logger.warning("Skipping %s: model creation failed — %s", name, exc)
                continue

            # Load checkpoint
            try:
                self._load_checkpoint(model, ckpt_path, device=device)
            except Exception as exc:
                logger.warning("Skipping %s: checkpoint loading failed — %s", name, exc)
                continue

            # Predictor
            predictor = Predictor(model, device=device)

            # Accumulate confusion matrix
            cm = ConfusionMatrix()
            with torch.inference_mode():
                for batch in data_loader:
                    if isinstance(batch, dict):
                        images = batch["image"]
                        masks = batch["mask"]
                    else:
                        images, masks = batch

                    images = images.to(device)
                    masks = masks.to(device)

                    preds = predictor.predict_batch(images)
                    cm.accumulate(preds.cpu(), masks.cpu(), self.config.binarize_threshold)

            metrics = cm.derive_metrics()
            results[name] = {
                "confusion": cm,
                "metrics": metrics,
                "checkpoint_path": str(ckpt_path),
            }

            logger.info(
                "%s — Dice=%.4f IoU=%.4f Sensitivity=%.4f Specificity=%.4f",
                name,
                metrics["dice"],
                metrics["iou"],
                metrics["sensitivity"],
                metrics["specificity"],
            )

        # --- Reporting (delegate) ---
        if self._reporter is not None and results:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            report_md = self._reporter.generate_markdown(results)
            (output_dir / "comparison_report.md").write_text(report_md, encoding="utf-8")

            justification_md = self._reporter.generate_clinical_justification(results)
            (output_dir / "clinical_justification.md").write_text(justification_md, encoding="utf-8")

        # --- Visualization (delegate) ---
        if self._visualizer is not None and results:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Per-model heatmaps
            for name, data in results.items():
                cm_data = data["confusion"]
                if isinstance(cm_data, ConfusionMatrix):
                    self._visualizer.plot_absolute(
                        cm_data, model_name=name,
                        output_path=output_dir / f"heatmap_absolute_{name}.png",
                    )
                    self._visualizer.plot_normalized(
                        cm_data, model_name=name,
                        output_path=output_dir / f"heatmap_normalized_{name}.png",
                    )

            # Comparison grid
            cms = {name: data["confusion"] for name, data in results.items()
                   if isinstance(data["confusion"], ConfusionMatrix)}
            if cms:
                self._visualizer.plot_comparison(
                    cms,
                    output_path=output_dir / "heatmap_comparison.png",
                )

        return results
