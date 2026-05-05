"""Core latent diffusion model scaffolding."""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LatentDiffusionConfig:
    image_resolution: int = 256
    latent_channels: int = 4
    conditioning_strategy: str = "cross_attention"
    condition_dim: int = 7
    hidden_channels: int = 64
    noise_std: float = 0.05


def build_model_name(config: LatentDiffusionConfig) -> str:
    """Return a stable name for experiment logs and checkpoints."""
    return f"ldm-{config.image_resolution}-{config.conditioning_strategy}"


class FiLMBlock(nn.Module):
    """A small convolutional block with feature-wise affine conditioning."""

    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.to_scale = nn.Linear(condition_dim, out_channels)
        self.to_shift = nn.Linear(condition_dim, out_channels)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.norm(h)
        scale = self.to_scale(condition).unsqueeze(-1).unsqueeze(-1)
        shift = self.to_shift(condition).unsqueeze(-1).unsqueeze(-1)
        h = h * (1.0 + scale) + shift
        return F.relu(h, inplace=True)


class ConditionalLatentScaffold(nn.Module):
    """Conditional denoising scaffold with multi-layer FiLM conditioning."""

    def __init__(self, config: LatentDiffusionConfig) -> None:
        super().__init__()
        self.condition_embedding = nn.Sequential(
            nn.Linear(config.condition_dim, config.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_channels, config.hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.input_proj = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.enc1 = FiLMBlock(32, 64, config.hidden_channels)
        self.down = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enc2 = FiLMBlock(64, config.hidden_channels, config.hidden_channels)
        self.bottleneck1 = FiLMBlock(
            config.hidden_channels, config.hidden_channels, config.hidden_channels
        )
        self.bottleneck2 = FiLMBlock(
            config.hidden_channels, config.hidden_channels, config.hidden_channels
        )
        self.up = nn.ConvTranspose2d(
            config.hidden_channels, config.hidden_channels, kernel_size=4, stride=2, padding=1
        )
        self.dec1 = FiLMBlock(config.hidden_channels, 64, config.hidden_channels)
        self.dec2 = FiLMBlock(64, 32, config.hidden_channels)
        self.output_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.severity_head = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, config.condition_dim),
        )

    def forward(
        self, image: torch.Tensor, severity_one_hot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.condition_embedding(severity_one_hot)
        h = F.relu(self.input_proj(image), inplace=True)
        h = self.enc1(h, condition)
        h = F.relu(self.down(h), inplace=True)
        h = self.enc2(h, condition)
        h = self.bottleneck1(h, condition)
        h = self.bottleneck2(h, condition)
        h = F.relu(self.up(h), inplace=True)
        h = self.dec1(h, condition)
        h = self.dec2(h, condition)
        reconstruction = self.output_head(h)
        severity_logits = self.severity_head(reconstruction)
        return reconstruction, severity_logits


def build_scaffold_model(config: LatentDiffusionConfig) -> nn.Module:
    """Construct the early conditional generative scaffold."""
    return ConditionalLatentScaffold(config)
