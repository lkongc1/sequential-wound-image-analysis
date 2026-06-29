"""Ensemble multiple models for inference (Pylance-clean)."""
from typing import List, Optional

import torch
import torch.nn as nn


class EnsemblePredictor:
    """Combine predictions from multiple models."""

    def __init__(self, models: List[nn.Module], device: Optional[str] = None):
        self.models = models
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        for m in self.models:
            m.to(self.device)
            m.eval()

    @torch.no_grad()
    def predict(self, image: torch.Tensor, method: str = "mean") -> torch.Tensor:
        """Ensemble prediction: 'mean' or 'max'."""
        image = image.to(self.device, non_blocking=True)
        preds: List[torch.Tensor] = []
        for m in self.models:
            with torch.cuda.amp.autocast():
                out = m(image)
            preds.append(torch.sigmoid(out))
        if method == "mean":
            return torch.mean(torch.stack(preds), dim=0)
        else:
            return torch.max(torch.stack(preds), dim=0)[0]
