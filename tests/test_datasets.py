import numpy as np
import pytest
import torch
from torchvision import transforms

from data import build_dataset
from data.dsprites import build_dsprites_dataset
from data.smallnorb import SmallNORBTensorDataset


def test_factory_rejects_missing_data_dirs_for_added_datasets():
    for name in ["celeba", "smallnorb", "dsprites", "dsprites_noisy"]:
        with pytest.raises(ValueError, match="requires data_dir"):
            build_dataset(name, {}, seed=0)


def test_smallnorb_tensor_dataset_resizes_images():
    dataset = SmallNORBTensorDataset(
        torch.zeros(2, 96, 96, dtype=torch.uint8),
        transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]),
    )

    assert dataset[0].shape == (1, 32, 32)


def test_dsprites_validation_split_and_noise_are_reproducible(tmp_path):
    images = np.zeros((10, 64, 64), dtype=np.uint8)
    images[:, 8:16, 8:16] = 1
    np.savez(tmp_path / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", imgs=images)
    config = {
        "data_dir": str(tmp_path),
        "split": "train",
        "image_size": 32,
        "validation_fraction": 0.2,
    }

    clean = build_dsprites_dataset(config, seed=7, noisy=False)
    noisy_first = build_dsprites_dataset(config, seed=7, noisy=True)
    noisy_second = build_dsprites_dataset(config, seed=7, noisy=True)

    assert len(clean) == 8
    assert clean[0][0].shape == (1, 32, 32)
    assert torch.equal(noisy_first[0][0], noisy_second[0][0])
    assert not torch.equal(clean[0][0], noisy_first[0][0])
