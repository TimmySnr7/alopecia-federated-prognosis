"""Train the first centralised conditional generative scaffold for Experiment 1."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import yaml

from data.datasets import ConditionalManifestDataset
from models.diffusion.latent_diffusion import (
    LatentDiffusionConfig,
    build_model_name,
    build_scaffold_model,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text())


def _build_loader(
    manifest_path: Path,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    class_values: list[int] | None,
    max_samples: int | None,
) -> DataLoader:
    dataset = ConditionalManifestDataset(
        manifest_path=manifest_path,
        image_size=image_size,
        class_values=class_values,
        max_samples=max_samples,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW | None,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, int]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        images = batch["image"].to(device)
        severity_one_hot = batch["severity_one_hot"].to(device)

        if training:
            optimizer.zero_grad()

        reconstructions = model(images, severity_one_hot)
        loss = criterion(reconstructions, images)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_examples += images.size(0)

    mean_loss = total_loss / total_examples if total_examples else 0.0
    return mean_loss, total_examples


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to exp01 configuration file.")
    parser.add_argument("train_manifest", type=Path)
    parser.add_argument("val_manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    _repo_root()
    experiment_config = _load_config(args.config)
    image_size = experiment_config["dataset"]["resolution"]
    learning_rate = experiment_config["training"]["learning_rate"]
    conditioning_strategy = experiment_config["model"]["conditioning"]

    train_loader = _build_loader(
        manifest_path=args.train_manifest,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=True,
        class_values=None,
        max_samples=args.max_train_samples,
    )
    train_class_values = train_loader.dataset.class_values
    val_loader = _build_loader(
        manifest_path=args.val_manifest,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=False,
        class_values=train_class_values,
        max_samples=args.max_val_samples,
    )

    condition_dim = len(train_class_values)
    model_config = LatentDiffusionConfig(
        image_resolution=image_size,
        conditioning_strategy=conditioning_strategy,
        condition_dim=condition_dim,
    )
    model = build_scaffold_model(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_examples = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_loss, val_examples = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
        )

        epoch_summary = {
            "epoch": epoch,
            "train_reconstruction_loss": train_loss,
            "val_reconstruction_loss": val_loss,
            "train_examples": train_examples,
            "val_examples": val_examples,
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary))

    summary = {
        "experiment": experiment_config["name"],
        "description": experiment_config["description"],
        "model_name": build_model_name(model_config),
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest),
        "device": str(device),
        "image_size": image_size,
        "conditioning_strategy": conditioning_strategy,
        "condition_classes": train_class_values,
        "train_sample_count": len(train_loader.dataset),
        "val_sample_count": len(val_loader.dataset),
        "history": history,
    }
    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
