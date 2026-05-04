"""Client-side scaffolding for local training and DP-SGD hooks."""

from dataclasses import dataclass


@dataclass
class ClientConfig:
    client_id: str
    local_epochs: int = 1
    dp_enabled: bool = True
