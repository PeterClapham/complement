import torch

from artifacts import save_reconstruction_grid
from data import SyntheticBinaryImageDataset
from models import VariationalGONGenerator


def test_save_reconstruction_grid_writes_image(tmp_path):
    model = VariationalGONGenerator(latent_dim=8, base_channels=4)
    dataset = SyntheticBinaryImageDataset(num_samples=8, image_size=32, channels=1, seed=3)

    path = save_reconstruction_grid(
        model=model,
        dataset=dataset,
        output_path=tmp_path / "grid.png",
        latent_dim=8,
        beta_inf=1.0,
        batch_size=4,
        device=torch.device("cpu"),
        nrow=2,
    )

    assert path.exists()
    assert path.stat().st_size > 0
