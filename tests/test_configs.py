from pathlib import Path

import yaml


def test_all_configs_have_name_and_mode() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "experiments" / "configs"
    for config_path in config_dir.glob("*.yaml"):
        config = yaml.safe_load(config_path.read_text())
        assert "name" in config
        assert "mode" in config
