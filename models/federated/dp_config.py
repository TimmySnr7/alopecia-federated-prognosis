"""Differential privacy configuration container."""

from dataclasses import dataclass


@dataclass
class DPConfig:
    epsilon: float = 3.0
    delta: float = 1e-5
    clipping_norm: float = 1.0
