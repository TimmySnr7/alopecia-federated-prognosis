"""Build pooled dataset indices and split manifests for experiment 1."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Any, Iterable

from data.registry import required_dataset_entries

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

ROMAN_STAGE_MAP = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
}


@dataclass(frozen=True)
class ManifestRecord:
    dataset_key: str
    image_path: str
    label_schema: str
    image_view: str
    severity_proxy_raw: str
    severity_proxy_value: str
    split: str


def _is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _is_mask_file(path: Path) -> bool:
    path_text = path.as_posix().lower()
    stem = path.stem.lower()
    return "mask" in stem or "/mask" in path_text


def _infer_view(path: Path) -> str:
    path_text = path.as_posix().lower()
    if "top-down" in path_text or "top_down" in path_text or "top.png" in path_text:
        return "top"
    if "front" in path_text:
        return "front"
    if "left" in path_text:
        return "left"
    if "right" in path_text:
        return "right"
    if "back" in path_text:
        return "back"
    return "unknown"


def _iter_image_paths(dataset_dir: Path) -> Iterable[Path]:
    for path in sorted(dataset_dir.rglob("*")):
        if _is_supported_image(path) and not _is_mask_file(path):
            yield path


def _extract_stage_token(path: Path) -> str:
    path_text = path.as_posix().lower()
    match = re.search(r"(norwood|ludwig|stage|grade)[-_ ]*([ivx]+|\d+)", path_text)
    if match:
        return match.group(2)

    digit_match = re.search(r"(?<!\d)([1-7])(?!\d)", path.stem.lower())
    if digit_match:
        return digit_match.group(1)

    roman_match = re.search(r"(?<![a-z])(i{1,3}|iv|v|vi{0,2}|vii)(?![a-z])", path.stem.lower())
    if roman_match:
        return roman_match.group(1)

    return ""


def _normalise_stage_value(token: str) -> str:
    if not token:
        return ""
    if token.isdigit():
        return token
    return str(ROMAN_STAGE_MAP.get(token.lower(), ""))


def _assign_splits(records: list[ManifestRecord], seed: int) -> list[ManifestRecord]:
    if not records:
        return []

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    total = len(shuffled)
    if total >= 3:
        test_count = max(1, int(round(total * 0.15)))
        val_count = max(1, int(round(total * 0.15)))
        if val_count + test_count >= total:
            val_count = 1
            test_count = 1
        train_count = total - val_count - test_count
    elif total == 2:
        train_count, val_count, test_count = 1, 0, 1
    else:
        train_count, val_count, test_count = 1, 0, 0

    train_cutoff = train_count
    val_cutoff = train_cutoff + val_count

    assigned: list[ManifestRecord] = []
    for index, record in enumerate(shuffled):
        if index < train_cutoff:
            split = "train"
        elif index < val_cutoff:
            split = "val"
        else:
            split = "test"
        assigned.append(
                ManifestRecord(
                    dataset_key=record.dataset_key,
                    image_path=record.image_path,
                    label_schema=record.label_schema,
                    image_view=record.image_view,
                    severity_proxy_raw=record.severity_proxy_raw,
                    severity_proxy_value=record.severity_proxy_value,
                    split=split,
                )
        )
    return assigned


def build_exp01_records(
    data_root: Path,
    registry: dict[str, Any],
    dataset_keys: list[str],
    seed: int,
) -> list[ManifestRecord]:
    """Scan raw dataset folders and create pooled manifest records."""
    entries = required_dataset_entries(registry, dataset_keys)
    records: list[ManifestRecord] = []

    for dataset_key, entry in entries.items():
        dataset_dir = data_root / "raw" / dataset_key
        if not dataset_dir.exists():
            continue

        label_schema = entry["label_schema"]["primary"]
        for image_path in _iter_image_paths(dataset_dir):
            token = _extract_stage_token(image_path)
            records.append(
                ManifestRecord(
                    dataset_key=dataset_key,
                    image_path=str(image_path.resolve()),
                    label_schema=label_schema,
                    image_view=_infer_view(image_path),
                    severity_proxy_raw=token,
                    severity_proxy_value=_normalise_stage_value(token),
                    split="",
                )
            )

    return _assign_splits(records, seed)


def write_manifest_csv(records: list[ManifestRecord], output_path: Path) -> None:
    """Write manifest records to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_key",
                "image_path",
                "label_schema",
                "image_view",
                "severity_proxy_raw",
                "severity_proxy_value",
                "split",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def filter_split(records: list[ManifestRecord], split: str) -> list[ManifestRecord]:
    """Return only the records matching the requested split."""
    return [record for record in records if record.split == split]
