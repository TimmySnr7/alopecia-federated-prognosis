"""Smoke-test the Experiment 1 manifest-backed image loader."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader

from data.datasets import ManifestImageDataset


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("manifest", type=Path, help="CSV manifest path to load.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    dataset = ManifestImageDataset(
        manifest_path=args.manifest,
        image_size=args.image_size,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Manifest: {args.manifest}")
    print(f"Samples loaded: {len(dataset)}")

    if len(dataset) == 0:
        print("No samples available.")
        return

    first_batch = next(iter(loader))
    print(f"Batch image shape: {tuple(first_batch['image'].shape)}")
    print(f"Batch severity shape: {tuple(first_batch['severity'].shape)}")
    print(f"Batch severities: {first_batch['severity'].tolist()}")

    counts = Counter(sample.severity_proxy_value for sample in dataset.samples)
    print(f"Severity distribution: {dict(sorted(counts.items()))}")
    print(f"Datasets present: {sorted({sample.dataset_key for sample in dataset.samples})}")


if __name__ == "__main__":
    main()
