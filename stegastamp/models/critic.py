from __future__ import annotations

import torch
import torch.nn as nn


class StegaStampCritic(nn.Module):
    """
    Critic trained with Wasserstein objective on clean vs encoded images.
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(base_channels * 8, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(image).flatten(1))
