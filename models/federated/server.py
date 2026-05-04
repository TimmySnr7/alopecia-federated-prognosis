"""Server-side scaffolding for Flower-based orchestration."""

from dataclasses import dataclass


@dataclass
class ServerConfig:
    rounds: int = 100
    strategy: str = "fedavg"
