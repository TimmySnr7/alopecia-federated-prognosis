"""Train Experiment 05's rebuilt neural severity scorer on QA-gated proxy pairs."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from experiments.scripts.train_exp03_proxy_residual_editor import (
    _condition_classes,
    _load_pair_rows,
    _prepare_proxy_pair,
    _split_rows,
)
from models.severity_grading.norwood_classifier import (
    NorwoodClassifierConfig,
    build_baseline_classifier,
)


class ProxySeverityDataset(Dataset[dict[str, Any]]):
    """Proxy targets plus source identity anchors for severity scorer rebuild."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        image_size: int,
        classes: list[int],
        include_source_identity: bool,
        augment: bool,
    ) -> None:
        self.rows = rows
        self.image_size = image_size
        self.classes = classes
        self.include_source_identity = include_source_identity
        self.augment = augment
        self.items: list[tuple[int, bool]] = [(index, False) for index in range(len(rows))]
        if include_source_identity:
            seen: set[str] = set()
            for index, row in enumerate(rows):
                if row["case_id"] in seen:
                    continue
                seen.add(row["case_id"])
                self.items.append((index, True))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index, identity = self.items[index]
        row = self.rows[row_index]
        target = int(float(row["source_severity" if identity else "target_severity"]))
        item = _prepare_proxy_pair(
            row=row,
            target_severity=target,
            image_size=self.image_size,
            classes=self.classes,
            augment=self.augment,
        )
        return {
            "image": item["source"] if identity else item["proxy_target"],
            "severity": torch.tensor(target, dtype=torch.long),
            "case_id": row["case_id"],
            "dataset_key": row["dataset_key"],
        }


def _label_map(classes: list[int]) -> dict[int, int]:
    return {label: index for index, label in enumerate(classes)}


def _sampling_weights(dataset: ProxySeverityDataset) -> list[float]:
    labels: list[int] = []
    for row_index, identity in dataset.items:
        row = dataset.rows[row_index]
        labels.append(int(float(row["source_severity" if identity else "target_severity"])))
    counts = Counter(labels)
    return [1.0 / counts[label] for label in labels]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW | None,
    criterion: nn.Module,
    device: torch.device,
    label_map: dict[int, int],
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    labels_all: list[int] = []
    predictions_all: list[int] = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = torch.tensor(
            [label_map[int(value)] for value in batch["severity"].tolist()],
            dtype=torch.long,
            device=device,
        )
        with torch.set_grad_enabled(training):
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        total_loss += float(loss.item()) * images.shape[0]
        total_correct += int((predictions == labels).sum().item())
        total_examples += images.shape[0]
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(predictions.detach().cpu().tolist())

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "examples": total_examples,
        "labels": labels_all,
        "predictions": predictions_all,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("proxy_pair_manifest", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--include-source-identity", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rows = _load_pair_rows(args.proxy_pair_manifest)
    classes = _condition_classes(rows)
    label_map = _label_map(classes)
    train_rows = _split_rows(rows, "train")
    val_rows = _split_rows(rows, "val")
    test_rows = _split_rows(rows, "test")

    train_dataset = ProxySeverityDataset(
        train_rows, args.image_size, classes, include_source_identity=args.include_source_identity, augment=True
    )
    val_dataset = ProxySeverityDataset(
        val_rows, args.image_size, classes, include_source_identity=True, augment=False
    )
    test_dataset = ProxySeverityDataset(
        test_rows, args.image_size, classes, include_source_identity=True, augment=False
    )
    sampler = WeightedRandomSampler(
        torch.tensor(_sampling_weights(train_dataset), dtype=torch.double),
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_baseline_classifier(
        NorwoodClassifierConfig(class_count=len(classes), pretrained=args.pretrained)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    history: list[dict[str, float]] = []
    best_val_macro_f1 = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, criterion, device, label_map)
        val_metrics = _run_epoch(model, val_loader, None, criterion, device, label_map)
        val_macro_f1 = f1_score(
            val_metrics["labels"], val_metrics["predictions"], average="macro", zero_division=0
        )
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_macro_f1,
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary))
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics = _run_epoch(model, val_loader, None, criterion, device, label_map)
    test_metrics = _run_epoch(model, test_loader, None, criterion, device, label_map)
    ordered = list(range(len(classes)))
    inverse_label_map = {index: label for label, index in label_map.items()}

    def split_summary(metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "macro_f1": f1_score(
                metrics["labels"], metrics["predictions"], average="macro", zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                metrics["labels"], metrics["predictions"], labels=ordered
            ).tolist(),
            "classification_report": classification_report(
                metrics["labels"],
                metrics["predictions"],
                labels=ordered,
                target_names=[str(inverse_label_map[index]) for index in ordered],
                zero_division=0,
                output_dict=True,
            ),
        }

    summary = {
        "experiment": "exp05_proxy_pair_severity_scorer",
        "proxy_pair_manifest": str(args.proxy_pair_manifest),
        "device": str(device),
        "image_size": args.image_size,
        "classes": classes,
        "label_map": label_map,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "test_examples": len(test_dataset),
        "include_source_identity": args.include_source_identity,
        "pretrained": args.pretrained,
        "history": history,
        "best_val_macro_f1": best_val_macro_f1,
        "val": split_summary(val_metrics),
        "test": split_summary(test_metrics),
        "checkpoint_path": str(args.checkpoint_path),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "backbone": "resnet18",
                "class_count": len(classes),
                "pretrained": args.pretrained,
            },
            "label_map": label_map,
            "image_size": args.image_size,
            "summary": summary,
        },
        args.checkpoint_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
