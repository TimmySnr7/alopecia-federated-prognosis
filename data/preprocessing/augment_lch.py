"""Placeholder for L-channel augmentation used in representation balancing."""

from dataclasses import dataclass


@dataclass
class LChannelAugmentationConfig:
    low_delta: int = 20
    high_delta: int = 40


def describe_augmentation(config: LChannelAugmentationConfig) -> str:
    """Return a human-readable augmentation description."""
    return f"L-channel reduction between {config.low_delta} and {config.high_delta} units."
