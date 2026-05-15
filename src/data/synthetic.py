"""Small synthetic datasets for pipeline tests."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticBinaryImageDataset(Dataset):
    """Deterministic 32x32 binary images for smoke-testing training code."""

    def __init__(
        self,
        num_samples: int = 16,
        image_size: int = 32,
        channels: int = 1,
        seed: int = 0,
    ) -> None:
        self.num_samples = num_samples
        generator = torch.Generator().manual_seed(seed)
        images = torch.rand(num_samples, channels, image_size, image_size, generator=generator)
        self.images = (images > 0.5).float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.images[index]
