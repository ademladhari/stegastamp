from __future__ import annotations

import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Div2KDetectorDataset(Dataset):
    """
    Builds detector training samples by compositing transformed StegaStamp crops
    into large background images (DIV2K-like pipeline described in paper Sec 4.3).
    """

    def __init__(
        self,
        backgrounds_dir: str,
        stamps_dir: str,
        out_size: int = 1024,
        min_stamp_scale: float = 0.2,
        max_stamp_scale: float = 0.6,
    ) -> None:
        self.background_paths = sorted([p for p in Path(backgrounds_dir).glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        self.stamp_paths = sorted([p for p in Path(stamps_dir).glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not self.background_paths:
            raise FileNotFoundError(f"No background images in {backgrounds_dir}")
        if not self.stamp_paths:
            raise FileNotFoundError(f"No stamp images in {stamps_dir}")
        self.out_size = out_size
        self.min_stamp_scale = min_stamp_scale
        self.max_stamp_scale = max_stamp_scale
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return max(len(self.background_paths), len(self.stamp_paths))

    def _random_stamp(self) -> Image.Image:
        stamp = Image.open(random.choice(self.stamp_paths)).convert("RGB")
        angle = random.uniform(-45, 45)
        stamp = TF.rotate(stamp, angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
        return stamp

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        bg = Image.open(self.background_paths[idx % len(self.background_paths)]).convert("RGB")
        bg = TF.resize(bg, [self.out_size, self.out_size], interpolation=transforms.InterpolationMode.BILINEAR)
        stamp = self._random_stamp()

        scale = random.uniform(self.min_stamp_scale, self.max_stamp_scale)
        stamp_size = int(self.out_size * scale)
        stamp = TF.resize(stamp, [stamp_size, stamp_size], interpolation=transforms.InterpolationMode.BILINEAR)

        max_xy = self.out_size - stamp_size - 1
        x0 = random.randint(0, max(0, max_xy))
        y0 = random.randint(0, max(0, max_xy))

        bg_t = self.to_tensor(bg)
        stamp_t = self.to_tensor(stamp)
        mask = (stamp_t.mean(dim=0, keepdim=True) > 0.02).float()
        composite = bg_t.clone()
        composite[:, y0 : y0 + stamp_size, x0 : x0 + stamp_size] = (
            composite[:, y0 : y0 + stamp_size, x0 : x0 + stamp_size] * (1.0 - mask) + stamp_t * mask
        )

        full_mask = torch.zeros(1, self.out_size, self.out_size, dtype=torch.float32)
        full_mask[:, y0 : y0 + stamp_size, x0 : x0 + stamp_size] = mask
        return composite, full_mask
