"""Core latent diffusion model scaffolding."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LatentDiffusionConfig:
    image_resolution: int = 256
    latent_channels: int = 4
    conditioning_strategy: str = "cross_attention"
    condition_dim: int = 7
    hidden_channels: int = 32


def build_model_name(config: LatentDiffusionConfig) -> str:
    """Return a stable name for experiment logs and checkpoints."""
    return f"ldm-{config.image_resolution}-{config.conditioning_strategy}"


class ConditionalLatentScaffold(nn.Module):
    """Tiny conditional image-to-image scaffold for early Experiment 1 training."""

    def __init__(self, config: LatentDiffusionConfig) -> None:
        super().__init__()
        self.condition_projection = nn.Linear(config.condition_dim, config.hidden_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + config.hidden_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, severity_one_hot: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = image.shape
        condition_map = self.condition_projection(severity_one_hot).view(batch_size, -1, 1, 1)
        condition_map = condition_map.expand(-1, -1, height, width)
        fused = torch.cat([image, condition_map], dim=1)
        latent = self.encoder(fused)
        return self.decoder(latent)


def build_scaffold_model(config: LatentDiffusionConfig) -> nn.Module:
    """Construct the early conditional generative scaffold."""
    return ConditionalLatentScaffold(config)
