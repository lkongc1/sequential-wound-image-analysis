"""Inference predictor with postprocessing."""
from typing import Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from src.metrics.segmentation import calculate_metrics


class Predictor:
    """Run inference on trained models."""

    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        return ckpt.get("dice", 0.0)

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Predict mask for a single image tensor."""
        image = image.to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast():
            output = self.model(image)
        return torch.sigmoid(output)

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Predict masks for a batch of images."""
        images = images.to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = self.model(images)
        return torch.sigmoid(outputs)

    def evaluate_dataset(self, dataset) -> Dict[str, float]:
        """Evaluate entire dataset and return metrics."""
        all_metrics = []
        for sample in tqdm(dataset, desc="Evaluating"):
            img = sample["image"].unsqueeze(0).to(self.device)
            msk = sample["mask"].unsqueeze(0)
            with torch.cuda.amp.autocast():
                pred = self.model(img)
            pred_prob = torch.sigmoid(pred).cpu()
            metrics = calculate_metrics(pred_prob, msk)
            all_metrics.append(metrics)
        final = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0].keys()}
        return final
