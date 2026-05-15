import torch

from data.mnist import _maybe_cache_dataset


class _TinyDataset:
    def __init__(self):
        self.images = [torch.zeros(1, 2, 2), torch.ones(1, 2, 2)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], index


def test_mnist_tensor_cache_preserves_transformed_images(tmp_path):
    dataset = _TinyDataset()

    first = _maybe_cache_dataset(dataset, tmp_path, split="train", image_size=2, binarize=False)
    second = _maybe_cache_dataset(dataset, tmp_path, split="train", image_size=2, binarize=False)

    assert torch.equal(first[0][0], dataset[0][0])
    assert torch.equal(second[1][0], dataset[1][0])
    assert len(list((tmp_path / "tensor_cache").glob("*.pt"))) == 1
