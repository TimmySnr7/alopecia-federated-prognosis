"""Smoke-test the conditional dataset path for the Experiment 1 generative scaffold."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader

from data.datasets import ConditionalManifestDataset


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("manifest", type=Path, help="Manifest CSV to load.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    dataset = ConditionalManifestDataset(
        manifest_path=args.manifest,
        image_size=args.image_size,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Manifest: {args.manifest}")
    print(f"Samples loaded: {len(dataset)}")
    print(f"Condition classes: {dataset.class_values}")

    if len(dataset) == 0:
        print("No samples available.")
        return

    first_batch = next(iter(loader))
    print(f"Batch image shape: {tuple(first_batch['image'].shape)}")
    print(f"Batch current severity shape: {tuple(first_batch['current_severity'].shape)}")
    print(f"Batch target severity shape: {tuple(first_batch['target_severity'].shape)}")
    print(f"Batch one-hot shape: {tuple(first_batch['severity_one_hot'].shape)}")
    print(f"Batch current severities: {first_batch['current_severity'].tolist()}")
    print(f"Batch target severities: {first_batch['target_severity'].tolist()}")
    print(f"First one-hot vector: {first_batch['severity_one_hot'][0].tolist()}")


if __name__ == "__main__":
    main()
