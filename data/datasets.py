"""Dataset loaders backed by experiment manifest CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import (
    ColorJitter,
    Compose,
    InterpolationMode,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)


@dataclass(frozen=True)
class ManifestSample:
    dataset_key: str
    image_path: str
    label_schema: str
    image_view: str
    severity_proxy_raw: str
    severity_proxy_value: int
    split: str


def _build_transform(image_size: int, augment: bool) -> Compose:
    transforms = [
        Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
    ]
    if augment:
        transforms.extend(
            [
                RandomHorizontalFlip(p=0.5),
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            ]
        )
    transforms.append(ToTensor())
    return Compose(transforms)


class ManifestImageDataset(Dataset[dict[str, Any]]):
    """Read image paths and severity labels from a manifest CSV."""

    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 256,
        augment: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.transform = _build_transform(image_size=image_size, augment=augment)
        self.samples = self._load_samples(max_samples=max_samples)

    def _load_samples(self, max_samples: int | None) -> list[ManifestSample]:
        rows: list[ManifestSample] = []
        with self.manifest_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_path = Path(row["image_path"])
                if not image_path.exists():
                    continue

                severity_raw = str(row["severity_proxy_value"]).strip()
                if not severity_raw:
                    continue

                try:
                    severity_value = int(float(severity_raw))
                except ValueError:
                    continue

                rows.append(
                    ManifestSample(
                        dataset_key=row["dataset_key"],
                        image_path=row["image_path"],
                        label_schema=row["label_schema"],
                        image_view=row["image_view"],
                        severity_proxy_raw=str(row["severity_proxy_raw"]),
                        severity_proxy_value=severity_value,
                        split=row["split"],
                    )
                )
                if max_samples is not None and len(rows) >= max_samples:
                    break
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image_tensor: Tensor = self.transform(image)
        severity = torch.tensor(sample.severity_proxy_value, dtype=torch.long)
        return {
            "image": image_tensor,
            "severity": severity,
            "dataset_key": sample.dataset_key,
            "image_path": sample.image_path,
            "label_schema": sample.label_schema,
            "image_view": sample.image_view,
            "split": sample.split,
        }


class ConditionalManifestDataset(Dataset[dict[str, Any]]):
    """Return images with severity-conditioning fields for generative scaffolds."""

    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 256,
        max_samples: int | None = None,
    ) -> None:
        self.base_dataset = ManifestImageDataset(
            manifest_path=manifest_path,
            image_size=image_size,
            augment=False,
            max_samples=max_samples,
        )
        self.samples = self.base_dataset.samples
        self.class_values = sorted({sample.severity_proxy_value for sample in self.samples})
        self.class_to_index = {value: index for index, value in enumerate(self.class_values)}

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base_dataset[index]
        severity_value = int(item["severity"].item())
        severity_index = self.class_to_index[severity_value]
        one_hot = torch.zeros(len(self.class_values), dtype=torch.float32)
        one_hot[severity_index] = 1.0

        return {
            "image": item["image"],
            "current_severity": item["severity"],
            "target_severity": item["severity"],
            "severity_one_hot": one_hot,
            "dataset_key": item["dataset_key"],
            "image_path": item["image_path"],
            "label_schema": item["label_schema"],
            "image_view": item["image_view"],
            "split": item["split"],
        }
