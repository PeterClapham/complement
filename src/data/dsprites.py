"""dSprites dataset construction."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms

from data.common import limit_dataset, split_train_validation

_URL = (
    "https://github.com/google-deepmind/dsprites-dataset/raw/master/"
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)


def build_dsprites_dataset(config: dict, seed: int, noisy: bool = False) -> Dataset:
    """Build dSprites; noisy mode adds deterministic uniform noise per image."""
    data_dir = config.get("data_dir")
    if data_dir is None:
        raise ValueError("dSprites config requires data_dir")

    split = str(config.get("split", "train"))
    image_size = int(config.get("image_size", 32))
    download = bool(config.get("download", False))
    validation_fraction = float(config.get("validation_fraction", 0.0))
    max_samples = config.get("max_samples")
    noise_scale = float(config.get("noise_scale", 0.1))
    dataset = _load_dataset(Path(data_dir), image_size, download, seed, noisy, noise_scale)
    if split == "test":
        raise ValueError("dSprites has no canonical test split; use train/val with validation_fraction")
    dataset = split_train_validation(dataset, split, validation_fraction, seed)
    return limit_dataset(dataset, max_samples)


def _load_dataset(
    root: Path,
    image_size: int,
    download: bool,
    seed: int,
    noisy: bool,
    noise_scale: float,
) -> Dataset:
    path = root / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    if not path.exists():
        if not download:
            raise RuntimeError("dSprites not found. Set download: true to fetch it.")
        root.mkdir(parents=True, exist_ok=True)
        urlretrieve(_URL, path)

    images = np.load(path)["imgs"]
    tensor = torch.from_numpy(images).unsqueeze(1).float()
    if image_size != 64:
        tensor = transforms.Resize(image_size)(tensor)
    if noisy:
        generator = torch.Generator().manual_seed(seed)
        tensor = (tensor + torch.rand(tensor.shape, generator=generator) * noise_scale).clamp(0.0, 1.0)
    return TensorDataset(tensor)
