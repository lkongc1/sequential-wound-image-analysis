"""Wound type classifier — EfficientNet-B3 with 4-channel masked-crop input."""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

from src.models.factory import register_model

logger = logging.getLogger(__name__)

# Number of output features from EfficientNet-B3 before classification head.
_EFFICIENTNET_B3_FEATURES: int = 1536
# Output channels of conv_stem in EfficientNet-B3.
_EFFICIENTNET_B3_STEM_OUT: int = 40


@register_model("wound_classifier")
class WoundClassifier(nn.Module):
    """EfficientNet-B3 classifier for 7 wound types.

    Accepts 4-channel input (RGB + binary mask). The first conv layer is
    expanded to accept the extra channel by copying green-channel weights
    from the pretrained checkpoint (design decision: green channel most
    relevant for medical images).

    Args:
        num_classes: Number of wound type classes (default 7).
        pretrained: If True, load ImageNet-pretrained weights via timm.
        freeze_backbone: If True, freeze all backbone parameters, only
            train the classification head.
        dropout: Dropout probability before classification head (default 0.4).

    Example:
        model = WoundClassifier(num_classes=7, pretrained=True)
        x = torch.randn(2, 4, 384, 384)
        logits = model(x)  # (2, 7)
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for WoundClassifier. "
                "Run: pip install timm"
            )

        # Create EfficientNet-B3 backbone
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,  # remove classification head
        )
        self.num_classes = num_classes

        # --- Expand conv_stem from 3 to 4 input channels ---
        old_conv: nn.Conv2d = self.backbone.conv_stem  # type: ignore[attr-defined]
        old_weight = old_conv.weight.data  # (out_ch, 3, kH, kW)

        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            # Copy RGB weights from pretrained
            new_conv.weight[:, :3, :, :] = old_weight
            # Copy green-channel (index 1) weights for the mask channel (index 3)
            new_conv.weight[:, 3, :, :] = old_weight[:, 1, :, :]

        self.backbone.conv_stem = new_conv  # type: ignore[attr-defined]

        # --- Replace classification head ---
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(_EFFICIENTNET_B3_FEATURES, num_classes),
        )

        # --- Optional backbone freezing ---
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self._log_init(pretrained, freeze_backbone)

    def _log_init(self, pretrained: bool, freeze_backbone: bool) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "WoundClassifier: num_classes=%d, pretrained=%s, "
            "freeze_backbone=%s, params=%d (trainable=%d)",
            self.num_classes, pretrained, freeze_backbone, total, trainable,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 4, H, W) — RGB + mask channel.

        Returns:
            Log-probabilities of shape (B, num_classes).
        """
        features = self.backbone(x)  # (B, 1536)
        logits = self.head(features)  # (B, num_classes)
        return torch.log_softmax(logits, dim=1)
