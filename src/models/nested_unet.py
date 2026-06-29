"""Nested U-Net (U-Net++) architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.factory import register_model


@register_model("nested_unet")
class NestedUNet(nn.Module):
    """Nested U-Net++ with dense skip connections."""

    def __init__(self, in_channels=3, out_channels=1, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        nb_filter = [32, 64, 128, 256, 512]

        self.inc = DoubleConvNest(in_channels, nb_filter[0])
        self.down1 = DownNest(nb_filter[0], nb_filter[1])
        self.down2 = DownNest(nb_filter[1], nb_filter[2])
        self.down3 = DownNest(nb_filter[2], nb_filter[3])
        self.down4 = DownNest(nb_filter[3], nb_filter[4])

        self.up4_3 = UpNest(nb_filter[4], nb_filter[3])
        self.up3_2 = UpNest(nb_filter[3], nb_filter[2])
        self.up2_1 = UpNest(nb_filter[2], nb_filter[1])
        self.up1_0 = UpNest(nb_filter[1], nb_filter[0])

        # Nested skip pathways
        self.conv4_3 = DoubleConvNest(nb_filter[3] + nb_filter[3], nb_filter[3])
        self.conv3_2 = DoubleConvNest(nb_filter[2] + nb_filter[2], nb_filter[2])
        self.conv2_1 = DoubleConvNest(nb_filter[1] + nb_filter[1], nb_filter[1])
        self.conv1_0 = DoubleConvNest(nb_filter[0] + nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4_up = self.up4_3(x5, x4)
        x4_conv = self.conv4_3(torch.cat([x4_up, x4], dim=1))

        x3_up = self.up3_2(x4_conv, x3)
        x3_conv = self.conv3_2(torch.cat([x3_up, x3], dim=1))

        x2_up = self.up2_1(x3_conv, x2)
        x2_conv = self.conv2_1(torch.cat([x2_up, x2], dim=1))

        x1_up = self.up1_0(x2_conv, x1)
        x1_conv = self.conv1_0(torch.cat([x1_up, x1], dim=1))

        return self.final(x1_conv)   # logits — la loss aplica sigmoid


class DoubleConvNest(nn.Module):
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


class DownNest(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConvNest(in_ch, out_ch))

    def forward(self, x):
        return self.maxpool_conv(x)


class UpNest(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvNest(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))
