"""Train a small severity classifier from Experiment 1 manifest CSV files."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
from typing import Iterable

from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data.datasets import ManifestImageDataset
from models.severity_grading.norwood_classifier import (
    NorwoodClassifierConfig,
    build_baseline_classifier,
)


def _infer_label_map(labels: Iterable[int]) -> dict[int, int]:
    unique = sorted(set(labels))
    return {label: index for index, label in enumerate(unique)}


def _build_loader(
    manifest_path: Path,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    max_samples: int | None,
) -> DataLoader:
    dataset = ManifestImageDataset(
        manifest_path=manifest_path,
        image_size=image_size,
        augment=augment,
        max_samples=max_samples,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _remap_labels(severity_tensor: torch.Tensor, label_map: dict[int, int]) -> torch.Tensor:
    remapped = [label_map[int(value)] for value in severity_tensor.tolist()]
    return torch.tensor(remapped, dtype=torch.long, device=severity_tensor.device)


def _class_weight_tensor(train_labels: list[int], label_map: dict[int, int], device: torch.device) -> torch.Tensor:
    counts = Counter(train_labels)
    total = sum(counts.values())
    weights = []
    for original_label, mapped_label in sorted(label_map.items(), key=lambda item: item[1]):
        count = counts[original_label]
        weights.append(total / (len(label_map) * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam | None,
    criterion: nn.Module,
    device: torch.device,
    label_map: dict[int, int],
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    all_predictions: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = _remap_labels(batch["severity"].to(device), label_map)

        if training:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if training:
            loss.backward()
            optimizer.step()

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += int((predictions == labels).sum().item())
        total_examples += images.size(0)
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    if total_examples == 0:
        return 0.0, 0.0, [], []

    return total_loss / total_examples, total_correct / total_examples, all_labels, all_predictions


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("train_manifest", type=Path)
    parser.add_argument("val_manifest", type=Path)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--class-weighting", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    train_loader = _build_loader(
        manifest_path=args.train_manifest,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        augment=args.augment,
        max_samples=args.max_train_samples,
    )
    val_loader = _build_loader(
        manifest_path=args.val_manifest,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        max_samples=args.max_val_samples,
    )

    train_labels = [sample.severity_proxy_value for sample in train_loader.dataset.samples]
    label_map = _infer_label_map(train_labels)
    class_count = len(label_map)

    config = NorwoodClassifierConfig(class_count=class_count, pretrained=args.pretrained)
    model = build_baseline_classifier(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    class_weights = (
        _class_weight_tensor(train_labels, label_map, device) if args.class_weighting else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    inverse_label_map = {mapped: original for original, mapped in label_map.items()}

    history: list[dict[str, float]] = []
    final_val_labels: list[int] = []
    final_val_predictions: list[int] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            label_map=label_map,
        )
        val_loss, val_acc, val_labels, val_predictions = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            label_map=label_map,
        )
        final_val_labels = val_labels
        final_val_predictions = val_predictions

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary))

    ordered_labels = sorted(inverse_label_map.keys())
    val_macro_f1 = (
        f1_score(final_val_labels, final_val_predictions, average="macro", zero_division=0)
        if final_val_labels
        else 0.0
    )
    val_confusion = (
        confusion_matrix(final_val_labels, final_val_predictions, labels=ordered_labels).tolist()
        if final_val_labels
        else []
    )
    val_classification_report = (
        classification_report(
            final_val_labels,
            final_val_predictions,
            labels=ordered_labels,
            target_names=[str(inverse_label_map[label]) for label in ordered_labels],
            zero_division=0,
            output_dict=True,
        )
        if final_val_labels
        else {}
    )

    summary = {
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest),
        "device": str(device),
        "pretrained": args.pretrained,
        "augment": args.augment,
        "class_weighting": args.class_weighting,
        "train_sample_count": len(train_loader.dataset),
        "val_sample_count": len(val_loader.dataset),
        "label_map": label_map,
        "train_label_distribution": dict(sorted(Counter(train_labels).items())),
        "val_macro_f1": val_macro_f1,
        "val_confusion_matrix": val_confusion,
        "val_classification_report": val_classification_report,
        "history": history,
    }
    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
