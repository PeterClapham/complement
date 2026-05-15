"""Variational Gradient Origin Network model components."""

from __future__ import annotations

import torch
from torch import nn


class VariationalGONGenerator(nn.Module):
    """Convolutional variational GON generator.

    The default architecture follows the small 32x32 grayscale generator from the
    reference Variational-GON example, with configurable latent size, feature width,
    and output channels.
    """

    def __init__(
        self,
        latent_dim: int = 48,
        base_channels: int = 32,
        output_channels: int = 1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.output_channels = output_channels

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 4, 4, 1, 0, bias=True),
            nn.BatchNorm2d(base_channels * 4),
            nn.ELU(),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(base_channels * 2),
            nn.ELU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=True),
            nn.BatchNorm2d(base_channels),
            nn.ELU(),
            nn.ConvTranspose2d(base_channels, output_channels, 4, 2, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, latent_origin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latent-origin vectors into images and variational statistics."""
        mu = self.fc_mu(latent_origin)
        logvar = self.fc_logvar(latent_origin)
        latent = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(latent.unsqueeze(-1).unsqueeze(-1))
        return reconstruction, mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from N(mu, sigma) in training mode, and return mu in eval mode."""
        if not self.training:
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Sample images from a standard normal latent distribution."""
        parameter = next(self.parameters())
        sample_device = torch.device(device) if device is not None else parameter.device
        latent = torch.randn(batch_size, self.latent_dim, 1, 1, device=sample_device)
        return self.decoder(latent)
