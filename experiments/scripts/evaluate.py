"""Load and display an evaluation configuration."""

from argparse import ArgumentParser
from pathlib import Path

import yaml


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    metrics = config.get("evaluation", {}).get("metrics", [])
    print(f"Loaded evaluation config: {config['name']} with metrics {metrics}")


if __name__ == "__main__":
    main()
