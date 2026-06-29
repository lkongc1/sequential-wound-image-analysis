"""Attention U-Net architecture for wound segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.factory import register_model


class AttentionGate(nn.Module):
    """Attention gate for Attention U-Net."""

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


@register_model("attention_unet")
class AttentionUNet(nn.Module):
    """Attention U-Net with attention gates."""

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder
        self.inc = DoubleConvAttn(in_channels, 32)
        self.down1 = DownAttn(32, 64)
        self.down2 = DownAttn(64, 128)
        self.down3 = DownAttn(128, 256)
        self.bottleneck = DoubleConvAttn(256, 256)
        # Decoder with attention
        self.up1 = UpAttnAttn(256, 128, use_attention=True)
        self.up2 = UpAttnAttn(128, 64, use_attention=True)
        self.up3 = UpAttnAttn(64, 32, use_attention=True)
        self.outc = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(self.bottleneck(x4), x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)   # logits — la loss aplica sigmoid


class DoubleConvAttn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownAttn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConvAttn(in_ch, out_ch))

    def forward(self, x):
        return self.maxpool_conv(x)


class UpAttnAttn(nn.Module):
    def __init__(self, in_ch, out_ch, use_attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvAttn(in_ch, out_ch)
        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionGate(in_ch // 2, in_ch // 2, in_ch // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.use_attention:
            x2 = self.attn(x1, x2)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))
