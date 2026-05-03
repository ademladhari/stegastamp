from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sample_perspective_corners(
    batch_size: int,
    device: torch.device,
    max_delta: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor]:
    src = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).repeat(batch_size, 1, 1)
    noise = (torch.rand(batch_size, 4, 2, device=device) * 2.0 - 1.0) * max_delta
    dst = torch.clamp(src + noise, 0.0, 1.0)
    return src, dst


def _build_motion_kernel(kernel_size: int, angle_rad: torch.Tensor, device: torch.device) -> torch.Tensor:
    center = (kernel_size - 1) / 2.0
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(kernel_size, device=device, dtype=torch.float32),
            torch.arange(kernel_size, device=device, dtype=torch.float32),
            indexing="ij",
        ),
        dim=-1,
    )
    coords = coords - center
    x, y = coords[..., 1], coords[..., 0]
    dist = torch.abs(-torch.sin(angle_rad) * x + torch.cos(angle_rad) * y)
    kernel = (dist < 0.5).float()
    kernel = kernel / (kernel.sum() + 1e-8)
    return kernel


def _random_motion_blur(x: torch.Tensor) -> torch.Tensor:
    b, c, _, _ = x.shape
    out = []
    for i in range(b):
        k = int(torch.randint(3, 8, (1,), device=x.device).item())
        angle = torch.rand(1, device=x.device) * math.pi
        kernel = _build_motion_kernel(k, angle, x.device)
        kernel = kernel.view(1, 1, k, k).repeat(c, 1, 1, 1)
        blurred = F.conv2d(x[i : i + 1], kernel, padding=k // 2, groups=c)
        out.append(blurred)
    return torch.cat(out, dim=0)


def _gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(3 * sigma))
    size = 2 * radius + 1
    coords = torch.arange(size, dtype=torch.float32, device=device) - radius
    g = torch.exp(-(coords**2) / (2 * sigma**2 + 1e-8))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def _random_defocus_blur(x: torch.Tensor) -> torch.Tensor:
    b, c, _, _ = x.shape
    out = []
    for i in range(b):
        sigma = float(torch.empty(1, device=x.device).uniform_(1.0, 3.0).item())
        kernel = _gaussian_kernel(sigma, x.device)
        k = kernel.shape[0]
        kernel = kernel.view(1, 1, k, k).repeat(c, 1, 1, 1)
        blurred = F.conv2d(x[i : i + 1], kernel, padding=k // 2, groups=c)
        out.append(blurred)
    return torch.cat(out, dim=0)


def _affine_color_transform(x: torch.Tensor) -> torch.Tensor:
    b = x.shape[0]
    # 1) hue shift via RGB channel offsets in [-0.1, 0.1]
    offsets = torch.empty(b, 3, 1, 1, device=x.device).uniform_(-0.1, 0.1)
    x = x + offsets

    # 2) random desaturation
    gray = x.mean(dim=1, keepdim=True)
    alpha = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.0, 1.0)
    x = alpha * x + (1.0 - alpha) * gray

    # 3) brightness/contrast affine mx+b
    m = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.5, 1.5)
    bias = torch.empty(b, 1, 1, 1, device=x.device).uniform_(-0.3, 0.3)
    x = m * x + bias
    return torch.clamp(x, 0.0, 1.0)


def _add_noise(x: torch.Tensor) -> torch.Tensor:
    b = x.shape[0]
    sigma = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.0, 0.2)
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, 0.0, 1.0)


def _differentiable_jpeg_proxy(x: torch.Tensor, quality_min: int = 50, quality_max: int = 100) -> torch.Tensor:
    """
    Lightweight differentiable JPEG proxy:
    - 8x8 block averaging to mimic block transform smoothing
    - piecewise quantization approximation near zero from paper Eq. (1)
    """
    b, c, h, w = x.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    unfold = F.unfold(x_pad, kernel_size=8, stride=8)
    unfold = unfold.view(b, c, 64, -1)
    means = unfold.mean(dim=2, keepdim=True)
    centered = unfold - means

    q = torch.empty(b, 1, 1, 1, device=x.device).uniform_(quality_min / 100.0, quality_max / 100.0)
    scale = (1.0 - q) * 2.0 + 0.1
    y = centered / scale
    quantized = torch.where(y.abs() < 0.5, y**3, y)
    restored = quantized * scale + means

    restored = restored.view(b, c * 64, -1)
    x_rec = F.fold(restored, output_size=(h + pad_h, w + pad_w), kernel_size=8, stride=8)
    x_rec = x_rec[..., :h, :w]
    return torch.clamp(x_rec, 0.0, 1.0)


class PhysicalPerturbationPipeline(nn.Module):
    """
    Paper Sec 3 perturbations:
    - perspective warp
    - motion / defocus blur
    - color manipulations
    - additive gaussian noise
    - differentiable JPEG approximation
    """

    def __init__(self, enable: bool = True) -> None:
        super().__init__()
        self.enable = enable

    def _perspective(self, x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        if strength <= 0.0:
            return x
        src, dst = _sample_perspective_corners(x.shape[0], x.device, max_delta=0.1 * strength)
        # Solve homography by 8-parameter DLT for each batch element.
        out = []
        for i in range(x.shape[0]):
            s = src[i]
            d = dst[i]
            A = []
            b = []
            for j in range(4):
                xs, ys = s[j]
                xd, yd = d[j]
                A.append(torch.tensor([xs, ys, 1, 0, 0, 0, -xd * xs, -xd * ys], device=x.device))
                A.append(torch.tensor([0, 0, 0, xs, ys, 1, -yd * xs, -yd * ys], device=x.device))
                b.append(xd)
                b.append(yd)
            A = torch.stack(A, dim=0)
            bvec = torch.stack(b, dim=0)
            h = torch.linalg.lstsq(A, bvec).solution
            H = torch.cat([h, torch.ones(1, device=x.device)]).view(3, 3)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, 1, x.shape[-2], device=x.device),
                torch.linspace(0, 1, x.shape[-1], device=x.device),
                indexing="ij",
            )
            ones = torch.ones_like(grid_x)
            pts = torch.stack([grid_x, grid_y, ones], dim=-1).view(-1, 3).T
            warped = torch.linalg.inv(H) @ pts
            warped = warped / (warped[2:3, :] + 1e-8)
            gx = warped[0, :].view(x.shape[-2], x.shape[-1]) * 2.0 - 1.0
            gy = warped[1, :].view(x.shape[-2], x.shape[-1]) * 2.0 - 1.0
            grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
            out.append(F.grid_sample(x[i : i + 1], grid, align_corners=True, padding_mode="border"))
        return torch.cat(out, dim=0)

    def forward(
        self,
        image: torch.Tensor,
        perspective_strength: float = 1.0,
        blur_strength: float = 1.0,
        color_strength: float = 1.0,
        noise_strength: float = 1.0,
        jpeg_strength: float = 1.0,
    ) -> torch.Tensor:
        if not self.enable:
            return image

        x = self._perspective(image, strength=perspective_strength)

        if blur_strength > 0:
            x = _random_motion_blur(x)
            x = _random_defocus_blur(x)

        if color_strength > 0:
            x = _affine_color_transform(x)

        if noise_strength > 0:
            x = _add_noise(x)

        if jpeg_strength > 0:
            x = _differentiable_jpeg_proxy(x)

        return x
