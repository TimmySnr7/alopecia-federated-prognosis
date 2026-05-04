"""Scaffold for checking label consistency across pooled datasets."""

from collections import Counter
from typing import Iterable


def label_counts(labels: Iterable[str]) -> Counter:
    """Return frequency counts for simple consistency checks."""
    return Counter(labels)
