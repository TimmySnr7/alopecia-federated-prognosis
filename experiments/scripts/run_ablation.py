"""Load and display an ablation configuration."""

from argparse import ArgumentParser
from pathlib import Path

import yaml


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    print(f"Loaded ablation config: {config['name']}")


if __name__ == "__main__":
    main()
