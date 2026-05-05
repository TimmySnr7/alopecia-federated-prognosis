"""Train the first centralised conditional generative scaffold for Experiment 1."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import yaml

from data.datasets import ConditionalManifestDataset
from models.diffusion.latent_diffusion import (
    LatentDiffusionConfig,
    build_model_name,
    build_scaffold_model,
)


class HybridReconstructionLoss(nn.Module):
    """Combine L1 and MSE losses to discourage blurry mean-image solutions."""

    def __init__(self, l1_weight: float = 1.0, mse_weight: float = 0.5) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def per_example(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = (prediction - target).abs().mean(dim=(1, 2, 3))
        mse = ((prediction - target) ** 2).mean(dim=(1, 2, 3))
        return self.l1_weight * l1 + self.mse_weight * mse

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.per_example(prediction, target).mean()


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
    target_mode: str,
    target_shift_probability: float,
    max_samples: int | None,
) -> DataLoader:
    dataset = ConditionalManifestDataset(
        manifest_path=manifest_path,
        image_size=image_size,
        class_values=class_values,
        target_mode=target_mode,
        target_shift_probability=target_shift_probability,
        max_samples=max_samples,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW | None,
    criterion: nn.Module,
    device: torch.device,
    noise_std: float,
    severity_loss_weight: float,
) -> tuple[float, float, float, int]:
    training = optimizer is not None
    model.train(training)

    total_reconstruction_loss = 0.0
    total_severity_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        images = batch["image"].to(device)
        severity_one_hot = batch["target_severity_one_hot"].to(device)
        target_indices = batch["target_severity_index"].to(device)
        identity_targets = batch["is_identity_target"].to(device=device, dtype=torch.float32)
        noisy_images = torch.clamp(images + torch.randn_like(images) * noise_std, 0.0, 1.0)

        with torch.set_grad_enabled(training):
            if training:
                optimizer.zero_grad()

            reconstructions, severity_logits = model(noisy_images, severity_one_hot)
            reconstruction_per_example = criterion.per_example(reconstructions, images)
            if identity_targets.sum().item() > 0:
                reconstruction_loss = (
                    reconstruction_per_example * identity_targets
                ).sum() / identity_targets.sum()
            else:
                reconstruction_loss = reconstruction_per_example.mean() * 0.0
            severity_loss = nn.functional.cross_entropy(severity_logits, target_indices)
            loss = reconstruction_loss + severity_loss_weight * severity_loss

            if training:
                loss.backward()
                optimizer.step()

        total_reconstruction_loss += reconstruction_loss.item() * images.size(0)
        total_severity_loss += severity_loss.item() * images.size(0)
        total_correct += (severity_logits.argmax(dim=1) == target_indices).sum().item()
        total_examples += images.size(0)

    mean_reconstruction_loss = total_reconstruction_loss / total_examples if total_examples else 0.0
    mean_severity_loss = total_severity_loss / total_examples if total_examples else 0.0
    severity_accuracy = total_correct / total_examples if total_examples else 0.0
    return mean_reconstruction_loss, mean_severity_loss, severity_accuracy, total_examples


def _save_epoch_samples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    noise_std: float,
    output_dir: Path,
    epoch: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = next(iter(loader), None)
    if batch is None:
        return

    model.eval()
    with torch.no_grad():
        images = batch["image"].to(device)
        severity_one_hot = batch["target_severity_one_hot"].to(device)
        noisy_images = torch.clamp(images + torch.randn_like(images) * noise_std, 0.0, 1.0)
        reconstructions, _ = model(noisy_images, severity_one_hot)

    sample_count = min(4, images.size(0))
    triptych = torch.cat(
        [
            images[:sample_count].cpu(),
            noisy_images[:sample_count].cpu(),
            reconstructions[:sample_count].cpu(),
        ],
        dim=0,
    )
    save_image(triptych, output_dir / f"epoch_{epoch:02d}_samples.png", nrow=sample_count)


def _save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    model_config: LatentDiffusionConfig,
    condition_classes: list[int],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "image_resolution": model_config.image_resolution,
                "latent_channels": model_config.latent_channels,
                "conditioning_strategy": model_config.conditioning_strategy,
                "condition_dim": model_config.condition_dim,
                "hidden_channels": model_config.hidden_channels,
                "noise_std": model_config.noise_std,
            },
            "condition_classes": condition_classes,
        },
        checkpoint_path,
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to exp01 configuration file.")
    parser.add_argument("train_manifest", type=Path)
    parser.add_argument("val_manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--sample-output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    _repo_root()
    experiment_config = _load_config(args.config)
    image_size = experiment_config["dataset"]["resolution"]
    learning_rate = experiment_config["training"]["learning_rate"]
    conditioning_strategy = experiment_config["model"]["conditioning"]
    noise_std = 0.05
    severity_loss_weight = 1.0
    train_target_shift_probability = 0.8

    train_loader = _build_loader(
        manifest_path=args.train_manifest,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=True,
        class_values=None,
        target_mode="sampled",
        target_shift_probability=train_target_shift_probability,
        max_samples=args.max_train_samples,
    )
    train_class_values = train_loader.dataset.class_values
    val_loader = _build_loader(
        manifest_path=args.val_manifest,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=False,
        class_values=train_class_values,
        target_mode="identity",
        target_shift_probability=0.0,
        max_samples=args.max_val_samples,
    )

    condition_dim = len(train_class_values)
    model_config = LatentDiffusionConfig(
        image_resolution=image_size,
        conditioning_strategy=conditioning_strategy,
        condition_dim=condition_dim,
        noise_std=noise_std,
    )
    model = build_scaffold_model(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = HybridReconstructionLoss()

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_reconstruction_loss, train_severity_loss, train_severity_accuracy, train_examples = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            noise_std=noise_std,
            severity_loss_weight=severity_loss_weight,
        )
        val_reconstruction_loss, val_severity_loss, val_severity_accuracy, val_examples = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            noise_std=noise_std,
            severity_loss_weight=severity_loss_weight,
        )

        epoch_summary = {
            "epoch": epoch,
            "train_reconstruction_loss": train_reconstruction_loss,
            "train_severity_loss": train_severity_loss,
            "train_severity_accuracy": train_severity_accuracy,
            "val_reconstruction_loss": val_reconstruction_loss,
            "val_severity_loss": val_severity_loss,
            "val_severity_accuracy": val_severity_accuracy,
            "train_examples": train_examples,
            "val_examples": val_examples,
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary))

        if args.sample_output_dir is not None:
            _save_epoch_samples(
                model=model,
                loader=val_loader,
                device=device,
                noise_std=noise_std,
                output_dir=args.sample_output_dir,
                epoch=epoch,
            )

        if args.checkpoint_path is not None:
            _save_checkpoint(
                model=model,
                checkpoint_path=args.checkpoint_path,
                model_config=model_config,
                condition_classes=train_class_values,
            )

    summary = {
        "experiment": experiment_config["name"],
        "description": experiment_config["description"],
        "model_name": build_model_name(model_config),
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest),
        "device": str(device),
        "image_size": image_size,
        "conditioning_strategy": conditioning_strategy,
        "loss_name": "l1_plus_half_mse",
        "severity_loss_name": "cross_entropy",
        "severity_loss_weight": severity_loss_weight,
        "noise_std": noise_std,
        "train_target_mode": "sampled",
        "train_target_shift_probability": train_target_shift_probability,
        "condition_classes": train_class_values,
        "train_sample_count": len(train_loader.dataset),
        "val_sample_count": len(val_loader.dataset),
        "sample_output_dir": str(args.sample_output_dir) if args.sample_output_dir else None,
        "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else None,
        "history": history,
    }
    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
