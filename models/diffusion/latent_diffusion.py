"""Core latent diffusion model scaffolding."""

from dataclasses import dataclass


@dataclass
class LatentDiffusionConfig:
    image_resolution: int = 256
    latent_channels: int = 4
    conditioning_strategy: str = "cross_attention"


def build_model_name(config: LatentDiffusionConfig) -> str:
    """Return a stable name for experiment logs and checkpoints."""
    return f"ldm-{config.image_resolution}-{config.conditioning_strategy}"
