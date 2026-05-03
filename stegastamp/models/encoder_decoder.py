from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = _conv_block(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.block(x)
        return self.pool(feat), feat


class _Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = _conv_block(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class MessagePreprocessor(nn.Module):
    """
    Paper Section 4.1:
    - message bits -> FC -> 50x50x3 tensor
    - upsampled to image resolution and concatenated with RGB image
    """

    def __init__(self, message_bits: int = 100) -> None:
        super().__init__()
        self.fc = nn.Linear(message_bits, 50 * 50 * 3)

    def forward(self, message: torch.Tensor, output_hw: int) -> torch.Tensor:
        x = self.fc(message)
        x = x.view(-1, 3, 50, 50)
        return F.interpolate(x, size=(output_hw, output_hw), mode="bilinear", align_corners=False)


@dataclass
class EncoderOutput:
    stegastamp: torch.Tensor
    residual: torch.Tensor


class StegaStampEncoder(nn.Module):
    """
    U-Net style residual encoder.
    Input: image RGB + upsampled message embedding.
    Output: residual, added to image to produce encoded StegaStamp.
    """

    def __init__(self, message_bits: int = 100, base_channels: int = 64, max_residual: float = 0.2) -> None:
        super().__init__()
        self.max_residual = max_residual
        self.message_pre = MessagePreprocessor(message_bits=message_bits)
        self.d1 = _Down(6, base_channels)
        self.d2 = _Down(base_channels, base_channels * 2)
        self.d3 = _Down(base_channels * 2, base_channels * 4)
        self.d4 = _Down(base_channels * 4, base_channels * 8)
        self.bottleneck = _conv_block(base_channels * 8, base_channels * 16)
        self.u4 = _Up(base_channels * 16, base_channels * 8, base_channels * 8)
        self.u3 = _Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.u2 = _Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.u1 = _Up(base_channels * 2, base_channels, base_channels)
        self.residual_head = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, image: torch.Tensor, message: torch.Tensor) -> EncoderOutput:
        msg = self.message_pre(message, output_hw=image.shape[-1])
        x = torch.cat([image, msg], dim=1)
        x, s1 = self.d1(x)
        x, s2 = self.d2(x)
        x, s3 = self.d3(x)
        x, s4 = self.d4(x)
        x = self.bottleneck(x)
        x = self.u4(x, s4)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        residual = torch.tanh(self.residual_head(x)) * self.max_residual
        stegastamp = torch.clamp(image + residual, 0.0, 1.0)
        return EncoderOutput(stegastamp=stegastamp, residual=residual)


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6),
        )
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.fc_loc(self.localization(x).flatten(1)).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode="border")


class StegaStampDecoder(nn.Module):
    """
    Decoder with STN front-end, outputting message logits.
    """

    def __init__(self, message_bits: int = 100, base_channels: int = 64) -> None:
        super().__init__()
        self.stn = SpatialTransformer(in_channels=3)
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(base_channels * 8, message_bits)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stn(image)
        x = self.features(x).flatten(1)
        return self.head(x)
