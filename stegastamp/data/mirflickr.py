from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class MirflickrDataset(Dataset):
    """
    Paper uses MIRFLICKR images resampled to 400x400.
    Expects ImageFolder-style root with at least one subfolder.
    """

    def __init__(self, root: str, image_size: int = 400) -> None:
        if not Path(root).exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        self.dataset = ImageFolder(
            root=root,
            transform=transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                ]
            ),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image, _ = self.dataset[idx]
        return image
