"""Load and display a federated experiment configuration."""

from argparse import ArgumentParser
from pathlib import Path

import yaml


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    rounds = config.get("federation", {}).get("rounds", "unknown")
    print(f"Loaded federated config: {config['name']} with {rounds} rounds")


if __name__ == "__main__":
    main()
