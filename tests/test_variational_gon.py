import pytest
import torch

from models import VariationalGONGenerator
from training import elbo_inf_loss, elbo_opt_loss, gon_training_step, vae_loss
from utils import DEFAULT_BETA_VALUES, ExperimentLogger, iter_beta_grid


def test_variational_gon_generator_forward_shapes():
    model = VariationalGONGenerator(latent_dim=8, base_channels=4, output_channels=1)
    latent_origin = torch.zeros(2, 8)

    reconstruction, mu, logvar = model(latent_origin)

    assert reconstruction.shape == (2, 1, 32, 32)
    assert mu.shape == (2, 8)
    assert logvar.shape == (2, 8)
    assert torch.all((0.0 <= reconstruction) & (reconstruction <= 1.0))


def test_variational_gon_generator_eval_uses_mu_for_reparameterization():
    model = VariationalGONGenerator(latent_dim=4, base_channels=4)
    model.eval()
    mu = torch.randn(3, 4)
    logvar = torch.randn(3, 4)

    assert torch.equal(model.reparameterize(mu, logvar), mu)


def test_variational_gon_generator_sample_shape():
    model = VariationalGONGenerator(latent_dim=8, base_channels=4, output_channels=1)

    samples = model.sample(batch_size=3, device="cpu")

    assert samples.shape == (3, 1, 32, 32)


def test_vae_loss_returns_total_reconstruction_and_kl_terms():
    reconstruction = torch.full((2, 1, 4, 4), 0.5)
    target = torch.ones(2, 1, 4, 4)
    mu = torch.zeros(2, 3)
    logvar = torch.zeros(2, 3)

    total, reconstruction_loss, kl_divergence = vae_loss(
        reconstruction,
        target,
        mu,
        logvar,
        kl_weight=0.5,
    )

    assert total.shape == ()
    assert reconstruction_loss.shape == ()
    assert kl_divergence.shape == ()
    assert torch.isclose(kl_divergence, torch.tensor(0.0))
    assert torch.isclose(total, reconstruction_loss)


def test_elbo_inf_and_elbo_opt_use_different_betas():
    reconstruction = torch.full((2, 1, 4, 4), 0.5)
    target = torch.ones(2, 1, 4, 4)
    mu = torch.ones(2, 3)
    logvar = torch.zeros(2, 3)

    inf_terms = elbo_inf_loss(reconstruction, target, mu, logvar, beta_inf=0.1)
    opt_terms = elbo_opt_loss(reconstruction, target, mu, logvar, beta_opt=10.0)

    assert inf_terms.beta == 0.1
    assert opt_terms.beta == 10.0
    assert opt_terms.loss > inf_terms.loss


def test_vae_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        vae_loss(
            torch.full((2, 1, 4, 4), 0.5),
            torch.ones(2, 1, 8, 8),
            torch.zeros(2, 3),
            torch.zeros(2, 3),
        )


def test_beta_grid_contains_all_phase_diagram_coordinates():
    grid = iter_beta_grid()

    assert len(grid) == 49
    assert grid[0] == {"beta_inf": 0.01, "beta_opt": 0.01}
    assert grid[-1] == {"beta_inf": 10.0, "beta_opt": 10.0}
    assert {entry["beta_inf"] for entry in grid} == set(DEFAULT_BETA_VALUES)
    assert {entry["beta_opt"] for entry in grid} == set(DEFAULT_BETA_VALUES)


def test_gon_training_step_uses_inner_and_outer_elbo_terms():
    model = VariationalGONGenerator(latent_dim=8, base_channels=4, output_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch = torch.rand(2, 1, 32, 32)

    metrics = gon_training_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        latent_dim=8,
        beta_inf=0.05,
        beta_opt=5.0,
    )

    assert metrics["beta_inf"] == 0.05
    assert metrics["beta_opt"] == 5.0
    assert metrics["elbo_inf_loss"] > 0.0
    assert metrics["elbo_opt_loss"] > 0.0


def test_experiment_logger_saves_model_state(tmp_path):
    model = VariationalGONGenerator(latent_dim=4, base_channels=4)
    logger = ExperimentLogger(config={}, seed=0, results_dir=tmp_path)

    model_path = logger.save_model(model)

    assert model_path == logger.run_dir / "model.pt"
    assert model_path.exists()
