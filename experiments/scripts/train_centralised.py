"""Validate and summarise a centralised experiment configuration."""

from argparse import ArgumentParser
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ValidationMessage:
    level: str
    text: str


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _required_keys() -> tuple[str, ...]:
    return (
        "name",
        "mode",
        "dataset",
        "conditioning_schema",
        "model",
        "training",
        "evaluation",
        "compute_environment",
        "hypotheses",
    )


def _validate_config(config: dict[str, Any], registry: dict[str, Any]) -> list[ValidationMessage]:
    messages: list[ValidationMessage] = []

    for key in _required_keys():
        if key not in config:
            messages.append(ValidationMessage("error", f"Missing top-level key: {key}"))

    dataset = config.get("dataset", {})
    included = dataset.get("included", [])
    registry_datasets = registry.get("datasets", {})
    for dataset_key in included:
        if dataset_key not in registry_datasets:
            messages.append(
                ValidationMessage("error", f"Dataset '{dataset_key}' is missing from the registry")
            )

    evaluation = config.get("evaluation", {})
    if evaluation.get("calibration_target") != "severity_class_proxy":
        messages.append(
            ValidationMessage(
                "warning",
                "Calibration target is not set to 'severity_class_proxy'; ECE/Brier semantics may drift.",
            )
        )

    domain_fid = evaluation.get("domain_fid", {})
    if "feature_extractor" not in domain_fid:
        messages.append(
            ValidationMessage("error", "Domain FID requires an explicit feature extractor definition")
        )

    model = config.get("model", {})
    if "initial_checkpoint" not in model:
        messages.append(
            ValidationMessage("warning", "No initial checkpoint specified; epoch count may be misleading.")
        )

    training = config.get("training", {})
    if training.get("epochs", 0) <= 0:
        messages.append(ValidationMessage("error", "Epoch count must be positive"))

    if training.get("batch_size", 0) <= 0:
        messages.append(ValidationMessage("error", "Batch size must be positive"))

    return messages


def _build_summary(config: dict[str, Any], registry: dict[str, Any], config_path: Path) -> dict[str, Any]:
    dataset = config["dataset"]
    registry_datasets = registry["datasets"]
    included = dataset["included"]
    return {
        "experiment": config["name"],
        "config_path": str(config_path),
        "objective": config["objective"]["type"],
        "claim_boundary": config["objective"]["claim_boundary"],
        "datasets": [
            {
                "key": key,
                "title": registry_datasets[key]["title"],
                "role": registry_datasets[key]["role_in_exp01"],
                "label_schema": registry_datasets[key]["label_schema"]["primary"],
                "limitation": registry_datasets[key]["key_limitation"],
            }
            for key in included
        ],
        "conditioning": config["conditioning_schema"],
        "checkpoint": config["model"]["initial_checkpoint"],
        "calibration_target": config["evaluation"]["calibration_target"],
        "domain_fid": config["evaluation"]["domain_fid"],
        "compute_environment": config["compute_environment"],
        "hypotheses": config["hypotheses"],
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--write-summary",
        type=Path,
        default=None,
        help="Optional path for a JSON summary of the validated experiment spec.",
    )
    args = parser.parse_args()
    repo_root = _resolve_repo_root()
    config_path = args.config.resolve()
    config = _load_yaml(config_path)

    registry_path = repo_root / config["dataset"]["registry"]
    registry = _load_yaml(registry_path)
    messages = _validate_config(config, registry)
    errors = [message for message in messages if message.level == "error"]

    print(f"Loaded centralised config: {config['name']}")
    print(f"Registry: {registry_path}")
    for message in messages:
        print(f"[{message.level.upper()}] {message.text}")

    if errors:
        raise SystemExit(1)

    summary = _build_summary(config, registry, config_path)
    print(json.dumps(summary, indent=2))

    if args.write_summary is not None:
        args.write_summary.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote summary to {args.write_summary}")


if __name__ == "__main__":
    main()
