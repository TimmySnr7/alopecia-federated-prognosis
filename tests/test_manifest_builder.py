from pathlib import Path

import yaml

from data.preprocessing.manifest_builder import build_exp01_records, filter_split
from data.registry import load_registry


def test_manifest_builder_creates_deterministic_splits(tmp_path: Path) -> None:
    data_root = tmp_path / "alopecia_public"
    raw_root = data_root / "raw"
    dataset_dir = raw_root / "unidatapro_men_hair_loss"
    dataset_dir.mkdir(parents=True)

    for name in [
        "norwood_1_a.jpg",
        "norwood_2_b.jpg",
        "stage_3_c.png",
        "sample_unknown.jpg",
        "mask_top.png",
    ]:
        (dataset_dir / name).write_text("placeholder")

    repo_root = Path(__file__).resolve().parents[1]
    registry = load_registry(repo_root / "data" / "dataset_registry.yaml")
    records = build_exp01_records(
        data_root=data_root,
        registry=registry,
        dataset_keys=["unidatapro_men_hair_loss"],
        seed=42,
    )

    assert len(records) == 4
    assert sorted({record.split for record in records}) == ["test", "train", "val"]
    assert sum(1 for record in records if record.severity_proxy_value) == 3
    assert all("mask" not in record.image_path.lower() for record in records)
    assert len(filter_split(records, "train")) == 2
    assert len(filter_split(records, "val")) == 1
    assert len(filter_split(records, "test")) == 1


def test_manifest_builder_uses_metadata_labels_for_tapakah68(tmp_path: Path) -> None:
    data_root = tmp_path / "alopecia_public"
    dataset_dir = data_root / "raw" / "tapakah68_bald_people"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True)

    image_name = "eu.sample.jpg"
    (images_dir / image_name).write_text("placeholder")
    (dataset_dir / "bald_people.csv").write_text(
        "Unnamed: 0,images,type\n"
        f"0,images/{image_name},type_6\n"
    )

    repo_root = Path(__file__).resolve().parents[1]
    registry = load_registry(repo_root / "data" / "dataset_registry.yaml")
    records = build_exp01_records(
        data_root=data_root,
        registry=registry,
        dataset_keys=["tapakah68_bald_people"],
        seed=42,
    )

    assert len(records) == 1
    assert records[0].severity_proxy_raw == "6"
    assert records[0].severity_proxy_value == "6"
    assert records[0].image_view == "unknown"
