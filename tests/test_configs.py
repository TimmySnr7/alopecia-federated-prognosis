from pathlib import Path

import yaml


def test_all_configs_have_name_and_mode() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "experiments" / "configs"
    for config_path in config_dir.glob("*.yaml"):
        config = yaml.safe_load(config_path.read_text())
        assert "name" in config
        assert "mode" in config


def test_exp01_has_required_protocol_fields() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "experiments" / "configs" / "exp01_centralised_baseline.yaml"
    registry_path = repo_root / "data" / "dataset_registry.yaml"
    config = yaml.safe_load(config_path.read_text())
    registry = yaml.safe_load(registry_path.read_text())

    assert config["objective"]["type"] == "proxy_prognostic_generation"
    assert config["evaluation"]["calibration_target"] == "severity_class_proxy"
    assert "feature_extractor" in config["evaluation"]["domain_fid"]
    assert "initial_checkpoint" in config["model"]
    assert "hardware" in config["compute_environment"]
    assert "primary" in config["hypotheses"]

    included = config["dataset"]["included"]
    for dataset_key in included:
        assert dataset_key in registry["datasets"]


def test_exp01_registry_keys_are_unique() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    registry_path = repo_root / "data" / "dataset_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text())
    keys = list(registry["datasets"].keys())
    assert len(keys) == len(set(keys))
