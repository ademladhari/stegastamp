from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_weight_map(height: int, width: int, device: torch.device) -> torch.Tensor:
    """
    Paper Sec 4.4: increase L2 weighting near image edges with cosine dropoff.
    """
    y = torch.linspace(0, 1, steps=height, device=device).view(height, 1)
    x = torch.linspace(0, 1, steps=width, device=device).view(1, width)
    dist_to_edge = torch.minimum(torch.minimum(x, 1 - x), torch.minimum(y, 1 - y))
    dist_to_edge = dist_to_edge / (dist_to_edge.max() + 1e-8)
    center_weight = 0.5 * (1 - torch.cos(dist_to_edge * math.pi))
    edge_boost = 1.0 + (1.0 - center_weight)
    return edge_boost.unsqueeze(0).unsqueeze(0)


class ResidualLoss(nn.Module):
    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        h, w = residual.shape[-2:]
        weights = edge_weight_map(h, w, residual.device)
        return (weights * residual.pow(2)).mean()


class LPIPSLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            import lpips  # type: ignore

            self.lpips = lpips.LPIPS(net="alex")
            self.available = True
        except Exception:
            self.lpips = None
            self.available = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.available and self.lpips is not None:
            # LPIPS expects range [-1, 1]
            return self.lpips(x * 2 - 1, y * 2 - 1).mean()
        return F.l1_loss(x, y)


class WassersteinCriticLoss(nn.Module):
    def forward(
        self,
        critic_real: torch.Tensor,
        critic_fake: torch.Tensor,
    ) -> torch.Tensor:
        return critic_fake.mean() - critic_real.mean()


def critic_hinge_regularizer(critic_out: torch.Tensor, sign: float) -> torch.Tensor:
    """Keep critic logits near [-1, 1] to avoid unbounded WGAN drift."""
    return torch.relu(1.0 - sign * critic_out).mean()


def clip_critic_weights(critic: nn.Module, clamp: float = 0.01) -> None:
    """Classic WGAN weight clipping for Lipschitz-like constraint."""
    with torch.no_grad():
        for param in critic.parameters():
            param.clamp_(-clamp, clamp)
