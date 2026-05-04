"""Build pooled train/val/test manifests for experiment 1."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import yaml

from data.preprocessing import build_exp01_records, filter_split, write_manifest_csv
from data.registry import load_registry


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to exp01 configuration YAML.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="External data root containing raw/, processed/, metadata/, and manifests/ directories.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    config = yaml.safe_load(args.config.read_text())
    registry = load_registry(repo_root / config["dataset"]["registry"])
    records = build_exp01_records(
        data_root=args.data_root,
        registry=registry,
        dataset_keys=config["dataset"]["included"],
        seed=config["seed"],
    )

    metadata_dir = args.data_root / "metadata"
    manifests_dir = args.data_root / "manifests"
    summary_path = manifests_dir / "exp01_manifest_summary.json"

    master_path = metadata_dir / "exp01_master_index.csv"
    train_path = manifests_dir / "exp01_train.csv"
    val_path = manifests_dir / "exp01_val.csv"
    test_path = manifests_dir / "exp01_test.csv"

    write_manifest_csv(records, master_path)
    write_manifest_csv(filter_split(records, "train"), train_path)
    write_manifest_csv(filter_split(records, "val"), val_path)
    write_manifest_csv(filter_split(records, "test"), test_path)

    summary = {
        "experiment": config["name"],
        "data_root": str(args.data_root.resolve()),
        "record_count": len(records),
        "train_count": len(filter_split(records, "train")),
        "val_count": len(filter_split(records, "val")),
        "test_count": len(filter_split(records, "test")),
        "datasets_present": sorted({record.dataset_key for record in records}),
        "view_counts": {
            view: sum(1 for record in records if record.image_view == view)
            for view in sorted({record.image_view for record in records})
        },
        "records_with_severity_proxy": sum(
            1 for record in records if record.severity_proxy_value
        ),
        "master_index": str(master_path.resolve()),
        "split_manifests": {
            "train": str(train_path.resolve()),
            "val": str(val_path.resolve()),
            "test": str(test_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
