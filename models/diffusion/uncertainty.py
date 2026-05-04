"""Uncertainty estimation helpers for prognosis sampling."""

from typing import Sequence


def sample_mean(values: Sequence[float]) -> float:
    """Compute a simple arithmetic mean for placeholder uncertainty summaries."""
    return sum(values) / len(values) if values else 0.0
