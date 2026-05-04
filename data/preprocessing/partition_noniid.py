"""Utilities for generating simulated non-IID client splits."""

from random import Random
from typing import Sequence


def partition_indices(indices: Sequence[int], client_count: int, seed: int = 42) -> dict[str, list[int]]:
    """Create a lightweight shuffled baseline partition for client simulation."""
    rng = Random(seed)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    partitions = {f"client_{idx+1}": [] for idx in range(client_count)}
    for offset, value in enumerate(shuffled):
        partitions[f"client_{(offset % client_count) + 1}"].append(value)
    return partitions
