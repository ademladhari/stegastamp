from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ks: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = ks // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=ks, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ContextPath(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.s1 = ConvBNReLU(in_channels, base_channels, stride=2)
        self.s2 = ConvBNReLU(base_channels, base_channels * 2, stride=2)
        self.s3 = ConvBNReLU(base_channels * 2, base_channels * 4, stride=2)
        self.s4 = ConvBNReLU(base_channels * 4, base_channels * 8, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.s1(x)
        x = self.s2(x)
        c8 = self.s3(x)
        c16 = self.s4(c8)
        return c8, c16


class SpatialPath(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 64) -> None:
        super().__init__()
        self.path = nn.Sequential(
            ConvBNReLU(in_channels, 64, ks=7, stride=2),
            ConvBNReLU(64, 64, stride=2),
            ConvBNReLU(64, 64, stride=2),
            ConvBNReLU(64, out_channels, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.path(x)


class FeatureFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 128) -> None:
        super().__init__()
        self.conv = ConvBNReLU(in_channels, out_channels, ks=1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, spatial_feat: torch.Tensor, context_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([spatial_feat, context_feat], dim=1)
        x = self.conv(x)
        a = self.attn(x)
        return x * a + x


class BiSeNetDetector(nn.Module):
    """
    Lightweight BiSeNet-style segmentation model for StegaStamp localization.
    Output is 1-channel mask logits.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.spatial = SpatialPath(in_channels=in_channels, out_channels=64)
        self.context = ContextPath(in_channels=in_channels, base_channels=base_channels)
        self.context_proj = ConvBNReLU(base_channels * 8, 64, ks=1)
        self.fusion = FeatureFusion(in_channels=64 + 64, out_channels=128)
        self.head = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sp = self.spatial(x)
        _, c16 = self.context(x)
        c16 = self.context_proj(c16)
        c16 = F.interpolate(c16, size=sp.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fusion(sp, c16)
        logits = self.head(fused)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
