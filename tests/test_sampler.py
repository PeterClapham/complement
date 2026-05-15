import torch
from torch.utils.data import TensorDataset

from training.experiment import EpochRandomSampler


def test_epoch_random_sampler_matches_seeded_randperm():
    dataset = TensorDataset(torch.arange(8))
    sampler = EpochRandomSampler(dataset, seed=3)

    sampler.set_epoch(2)

    generator = torch.Generator().manual_seed(5)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    expected = torch.randperm(8, generator=generator).tolist()
    assert list(sampler) == expected
