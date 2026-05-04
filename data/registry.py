"""Helpers for loading and querying the dataset registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_registry(registry_path: Path) -> dict[str, Any]:
    """Load the YAML dataset registry from disk."""
    return yaml.safe_load(registry_path.read_text())


def dataset_entries(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return registry entries keyed by dataset identifier."""
    return registry.get("datasets", {})


def required_dataset_entries(
    registry: dict[str, Any], dataset_keys: list[str]
) -> dict[str, dict[str, Any]]:
    """Return the registry entries for the requested dataset keys."""
    entries = dataset_entries(registry)
    missing = [key for key in dataset_keys if key not in entries]
    if missing:
        raise KeyError(f"Missing dataset registry entries: {', '.join(missing)}")
    return {key: entries[key] for key in dataset_keys}
