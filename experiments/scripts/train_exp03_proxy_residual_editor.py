"""Train Experiment 03's QA-gated proxy residual editor."""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageOps
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, Resize, ToTensor

from experiments.scripts.run_exp01_plausible_proxy_sweep import (
    _add_label,
    _build_ellipse_mask,
    _expected_severity,
    _make_proxy_edit,
    _monotonic_fraction,
    _to_pil,
)
from experiments.scripts.run_exp01_proxy_batch_eval import _load_scorer, _visual_delta_metrics


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _load_pair_rows(pair_manifest: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with pair_manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not _parse_bool(row.get("auto_pass", "false")):
                continue
            image_path = Path(row["image_path"])
            if not image_path.exists():
                continue
            rows.append(row)
    return rows


def _split_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == split]


def _unique_case_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    cases: dict[str, dict[str, str]] = {}
    for row in rows:
        cases.setdefault(row["case_id"], row)
    return list(cases.values())


def _condition_classes(rows: list[dict[str, str]]) -> list[int]:
    values = {
        int(float(row["source_severity"]))
        for row in rows
    } | {
        int(float(row["target_severity"]))
        for row in rows
    }
    return sorted(values)


def _one_hot(value: int, classes: list[int]) -> torch.Tensor:
    vector = torch.zeros(len(classes), dtype=torch.float32)
    vector[classes.index(value)] = 1.0
    return vector


def _condition_vector(
    source_severity: int,
    target_severity: int,
    classes: list[int],
) -> torch.Tensor:
    min_severity = min(classes)
    max_severity = max(classes)
    severity_range = max(max_severity - min_severity, 1)
    scalars = torch.tensor(
        [
            (source_severity - min_severity) / severity_range,
            (target_severity - min_severity) / severity_range,
            (target_severity - source_severity) / severity_range,
        ],
        dtype=torch.float32,
    )
    return torch.cat(
        [
            _one_hot(source_severity, classes),
            _one_hot(target_severity, classes),
            scalars,
        ]
    )


def _load_proxy_parameters(row: dict[str, str]) -> dict[str, Any]:
    return json.loads(row["proxy_parameters"])


def _load_ellipse(row: dict[str, str]) -> list[float]:
    return [float(value) for value in json.loads(row["ellipse_mask"])]


def _prepare_proxy_pair(
    row: dict[str, str],
    target_severity: int,
    image_size: int,
    classes: list[int],
    augment: bool,
) -> dict[str, Any]:
    source_severity = int(float(row["source_severity"]))
    proxy_parameters = _load_proxy_parameters(row)
    ellipse = _load_ellipse(row)

    image = Image.open(row["image_path"]).convert("RGB")
    resized = Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)(image)

    if augment and random.random() < 0.5:
        resized = ImageOps.mirror(resized)
        ellipse = [1.0 - ellipse[0], ellipse[1], ellipse[2], ellipse[3]]

    blurred = resized.filter(
        ImageFilter.GaussianBlur(radius=float(proxy_parameters["texture_blur_radius"]))
    )
    source = ToTensor()(resized).unsqueeze(0)
    blurred_tensor = ToTensor()(blurred).unsqueeze(0)
    mask = _build_ellipse_mask(
        ellipse,
        image_size=image_size,
        device=torch.device("cpu"),
        blur_radius=float(proxy_parameters["mask_blur_radius"]),
    )
    proxy_target = _make_proxy_edit(
        source=source,
        blurred=blurred_tensor,
        mask=mask,
        target_severity=target_severity,
        source_severity=float(source_severity),
        min_severity=min(classes),
        max_severity=max(classes),
        hair_enhancement_strength=float(proxy_parameters["hair_enhancement_strength"]),
        hair_suppression_strength=float(proxy_parameters["hair_suppression_strength"]),
        smoothing_strength=float(proxy_parameters["smoothing_strength"]),
        tone_shift_strength=float(proxy_parameters["tone_shift_strength"]),
    )

    return {
        "source": source[0],
        "proxy_target": proxy_target[0],
        "mask": mask[0],
        "texture": torch.cat(
            [
                blurred_tensor[0],
                (blurred_tensor[0] - source[0]).clamp(min=0.0),
            ],
            dim=0,
        ),
        "condition": _condition_vector(source_severity, target_severity, classes),
        "source_severity": torch.tensor(source_severity, dtype=torch.long),
        "target_severity": torch.tensor(target_severity, dtype=torch.long),
        "case_id": row["case_id"],
        "dataset_key": row["dataset_key"],
        "split": row["split"],
        "image_path": row["image_path"],
    }


class ProxyPairDataset(Dataset[dict[str, Any]]):
    """Generate proxy targets from QA-gated pair rows."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        image_size: int,
        classes: list[int],
        augment: bool,
    ) -> None:
        self.rows = rows
        self.image_size = image_size
        self.classes = classes
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return _prepare_proxy_pair(
            row=row,
            target_severity=int(float(row["target_severity"])),
            image_size=self.image_size,
            classes=self.classes,
            augment=self.augment,
        )


class MaskedResidualEditor(nn.Module):
    """Small mask-constrained residual editor conditioned on source and target severity."""

    def __init__(
        self,
        condition_dim: int,
        hidden_channels: int = 32,
        residual_scale: float = 0.35,
        texture_channels: int = 0,
    ) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.texture_channels = texture_channels
        input_channels = 3 + 1 + texture_channels + condition_dim
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(hidden_channels * 6, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        source: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor,
        texture: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition_map = condition[:, :, None, None].expand(
            -1, -1, source.shape[-2], source.shape[-1]
        )
        inputs = [source, mask]
        if self.texture_channels:
            if texture is None:
                raise ValueError("Texture channels are enabled but no texture tensor was given.")
            inputs.append(texture)
        inputs.append(condition_map)
        h = torch.cat(inputs, dim=1)
        h1 = self.enc1(h)
        h2 = self.enc2(F.avg_pool2d(h1, kernel_size=2))
        h3 = self.enc3(F.avg_pool2d(h2, kernel_size=2))
        b = self.bottleneck(h3)
        u2 = F.interpolate(b, size=h2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, h2], dim=1))
        u1 = F.interpolate(d2, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, h1], dim=1))
        residual = self.output_head(d1)
        return torch.clamp(source + self.residual_scale * residual * mask, 0.0, 1.0)


def _masked_mean(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (delta * mask).sum() / (mask.sum() * delta.shape[1] + 1e-8)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
    unmasked_loss_weight: float,
    proxy_loss_weight: float,
    masked_proxy_loss_weight: float,
    delta_loss_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    count = 0

    for batch in loader:
        source = batch["source"].to(device)
        proxy_target = batch["proxy_target"].to(device)
        mask = batch["mask"].to(device)
        texture = batch["texture"].to(device)
        condition = batch["condition"].to(device)

        with torch.set_grad_enabled(training):
            if training:
                optimizer.zero_grad()
            prediction = model(source, mask, condition, texture)
            proxy_delta = (prediction - proxy_target).abs()
            source_delta = (prediction - source).abs()
            residual_delta = ((prediction - source) - (proxy_target - source)).abs()
            proxy_l1 = proxy_delta.mean()
            masked_proxy_l1 = _masked_mean(proxy_delta, mask)
            masked_delta_l1 = _masked_mean(residual_delta, mask)
            unmasked_source_delta = _masked_mean(source_delta, 1.0 - mask)
            loss = (
                proxy_loss_weight * proxy_l1
                + masked_proxy_loss_weight * masked_proxy_l1
                + delta_loss_weight * masked_delta_l1
                + unmasked_loss_weight * unmasked_source_delta
            )
            if training:
                loss.backward()
                optimizer.step()

        batch_size = source.shape[0]
        count += batch_size
        totals["loss"] += float(loss.item()) * batch_size
        totals["proxy_l1"] += float(proxy_l1.item()) * batch_size
        totals["masked_proxy_l1"] += float(masked_proxy_l1.item()) * batch_size
        totals["masked_delta_l1"] += float(masked_delta_l1.item()) * batch_size
        totals["unmasked_source_delta"] += float(unmasked_source_delta.item()) * batch_size

    return {key: value / max(count, 1) for key, value in totals.items()} | {"examples": count}


def _score_image(
    scorer: nn.Module,
    labels: list[int],
    image: torch.Tensor,
) -> dict[str, float | int]:
    logits = scorer(image)
    probabilities = torch.softmax(logits[0], dim=0)
    predicted_index = int(probabilities.argmax().item())
    return {
        "predicted_severity": int(labels[predicted_index]),
        "predicted_confidence": float(probabilities[predicted_index].item()),
        "expected_severity": _expected_severity(probabilities, labels),
    }


def _case_passes(
    monotonic_fraction: float,
    expected_span: float,
    max_unmasked_delta: float,
    mask_area_fraction: float,
    min_monotonic_fraction: float,
    min_expected_span: float,
    max_unmasked_delta_threshold: float,
    mask_area_min: float,
    mask_area_max: float,
) -> bool:
    return (
        monotonic_fraction >= min_monotonic_fraction
        and expected_span >= min_expected_span
        and max_unmasked_delta <= max_unmasked_delta_threshold
        and mask_area_min <= mask_area_fraction <= mask_area_max
    )


def _compose_rows(rows: list[list[Image.Image]]) -> Image.Image:
    row_width = max(sum(panel.width for panel in row) for row in rows)
    height = sum(row[0].height for row in rows)
    canvas = Image.new("RGB", (row_width, height), (255, 255, 255))
    y = 0
    for row in rows:
        x = 0
        for panel in row:
            canvas.paste(panel, (x, y))
            x += panel.width
        y += row[0].height
    return canvas


def _evaluate_cases(
    model: nn.Module,
    case_rows: list[dict[str, str]],
    image_size: int,
    classes: list[int],
    scorer: nn.Module,
    scorer_labels: list[int],
    device: torch.device,
    output_contact_sheet: Path | None,
    split_name: str,
    min_monotonic_fraction: float,
    min_expected_span: float,
    max_unmasked_delta_threshold: float,
    mask_area_min: float,
    mask_area_max: float,
) -> dict[str, Any]:
    model.eval()
    case_summaries: list[dict[str, Any]] = []
    contact_rows: list[list[Image.Image]] = []

    with torch.no_grad():
        for row in case_rows:
            source_severity = int(float(row["source_severity"]))
            source_item = _prepare_proxy_pair(
                row=row,
                target_severity=source_severity,
                image_size=image_size,
                classes=classes,
                augment=False,
            )
            source = source_item["source"].unsqueeze(0).to(device)
            mask = source_item["mask"].unsqueeze(0).to(device)
            source_score = _score_image(scorer, scorer_labels, source)

            proxy_expected: list[float] = []
            proxy_predicted: list[int] = []
            learned_expected: list[float] = []
            learned_predicted: list[int] = []
            learned_max_unmasked_delta = 0.0
            learned_max_masked_delta = 0.0
            learned_proxy_l1_values: list[float] = []
            proxy_panels: list[Image.Image] = [
                _add_label(
                    _to_pil(source[0]),
                    [
                        f"{split_name} source",
                        f"sev {source_severity}",
                        row["dataset_key"][:24],
                    ],
                )
            ]
            learned_panels: list[Image.Image] = [
                _add_label(
                    _to_pil(source[0]),
                    [
                        "learned",
                        f"source exp {source_score['expected_severity']:.2f}",
                    ],
                )
            ]

            for target_severity in classes:
                item = _prepare_proxy_pair(
                    row=row,
                    target_severity=target_severity,
                    image_size=image_size,
                    classes=classes,
                    augment=False,
                )
                proxy_target = item["proxy_target"].unsqueeze(0).to(device)
                texture = item["texture"].unsqueeze(0).to(device)
                condition = item["condition"].unsqueeze(0).to(device)
                learned = model(source, mask, condition, texture)

                proxy_score = _score_image(scorer, scorer_labels, proxy_target)
                learned_score = _score_image(scorer, scorer_labels, learned)
                proxy_expected.append(float(proxy_score["expected_severity"]))
                proxy_predicted.append(int(proxy_score["predicted_severity"]))
                learned_expected.append(float(learned_score["expected_severity"]))
                learned_predicted.append(int(learned_score["predicted_severity"]))

                learned_metrics = _visual_delta_metrics(source, learned, mask)
                learned_max_unmasked_delta = max(
                    learned_max_unmasked_delta,
                    learned_metrics["unmasked_mean_absolute_delta"],
                )
                learned_max_masked_delta = max(
                    learned_max_masked_delta,
                    learned_metrics["masked_mean_absolute_delta"],
                )
                learned_proxy_l1_values.append(float((learned - proxy_target).abs().mean().item()))

                proxy_panels.append(
                    _add_label(
                        _to_pil(proxy_target[0]),
                        [
                            f"proxy t{target_severity}",
                            f"pred {proxy_score['predicted_severity']}",
                            f"exp {proxy_score['expected_severity']:.2f}",
                        ],
                    )
                )
                learned_panels.append(
                    _add_label(
                        _to_pil(learned[0]),
                        [
                            f"learned t{target_severity}",
                            f"pred {learned_score['predicted_severity']}",
                            f"exp {learned_score['expected_severity']:.2f}",
                        ],
                    )
                )

            learned_monotonic = _monotonic_fraction(learned_expected)
            proxy_monotonic = _monotonic_fraction(proxy_expected)
            learned_span = max(learned_expected) - min(learned_expected)
            proxy_span = max(proxy_expected) - min(proxy_expected)
            mask_area_fraction = float(mask.mean().item())
            learned_auto_pass = _case_passes(
                monotonic_fraction=learned_monotonic,
                expected_span=learned_span,
                max_unmasked_delta=learned_max_unmasked_delta,
                mask_area_fraction=mask_area_fraction,
                min_monotonic_fraction=min_monotonic_fraction,
                min_expected_span=min_expected_span,
                max_unmasked_delta_threshold=max_unmasked_delta_threshold,
                mask_area_min=mask_area_min,
                mask_area_max=mask_area_max,
            )
            proxy_auto_pass = _case_passes(
                monotonic_fraction=proxy_monotonic,
                expected_span=proxy_span,
                max_unmasked_delta=0.0,
                mask_area_fraction=mask_area_fraction,
                min_monotonic_fraction=min_monotonic_fraction,
                min_expected_span=min_expected_span,
                max_unmasked_delta_threshold=max_unmasked_delta_threshold,
                mask_area_min=mask_area_min,
                mask_area_max=mask_area_max,
            )
            case_summaries.append(
                {
                    "case_id": row["case_id"],
                    "dataset_key": row["dataset_key"],
                    "split": row["split"],
                    "source_severity": source_severity,
                    "source_expected_severity": source_score["expected_severity"],
                    "proxy_expected_severities": proxy_expected,
                    "learned_expected_severities": learned_expected,
                    "proxy_predicted_severities": proxy_predicted,
                    "learned_predicted_severities": learned_predicted,
                    "proxy_monotonic_fraction": proxy_monotonic,
                    "learned_monotonic_fraction": learned_monotonic,
                    "proxy_expected_span": proxy_span,
                    "learned_expected_span": learned_span,
                    "learned_mean_l1_to_proxy": mean(learned_proxy_l1_values),
                    "learned_max_masked_mean_absolute_delta": learned_max_masked_delta,
                    "learned_max_unmasked_mean_absolute_delta": learned_max_unmasked_delta,
                    "mask_area_fraction": mask_area_fraction,
                    "proxy_auto_pass": proxy_auto_pass,
                    "learned_auto_pass": learned_auto_pass,
                }
            )
            contact_rows.extend([proxy_panels, learned_panels])

    if output_contact_sheet is not None and contact_rows:
        output_contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        _compose_rows(contact_rows).save(output_contact_sheet)

    learned_pass_count = sum(case["learned_auto_pass"] for case in case_summaries)
    proxy_pass_count = sum(case["proxy_auto_pass"] for case in case_summaries)
    return {
        "split": split_name,
        "case_count": len(case_summaries),
        "learned_auto_pass_count": learned_pass_count,
        "learned_auto_pass_rate": learned_pass_count / max(len(case_summaries), 1),
        "proxy_auto_pass_count": proxy_pass_count,
        "proxy_auto_pass_rate": proxy_pass_count / max(len(case_summaries), 1),
        "mean_learned_monotonic_fraction": mean(
            [case["learned_monotonic_fraction"] for case in case_summaries]
        )
        if case_summaries
        else 0.0,
        "mean_proxy_monotonic_fraction": mean(
            [case["proxy_monotonic_fraction"] for case in case_summaries]
        )
        if case_summaries
        else 0.0,
        "mean_learned_expected_span": mean(
            [case["learned_expected_span"] for case in case_summaries]
        )
        if case_summaries
        else 0.0,
        "mean_proxy_expected_span": mean([case["proxy_expected_span"] for case in case_summaries])
        if case_summaries
        else 0.0,
        "mean_learned_l1_to_proxy": mean(
            [case["learned_mean_l1_to_proxy"] for case in case_summaries]
        )
        if case_summaries
        else 0.0,
        "mean_learned_max_unmasked_delta": mean(
            [case["learned_max_unmasked_mean_absolute_delta"] for case in case_summaries]
        )
        if case_summaries
        else 0.0,
        "cases": case_summaries,
        "contact_sheet": str(output_contact_sheet) if output_contact_sheet else None,
    }


def _save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    classes: list[int],
    image_size: int,
    condition_dim: int,
    residual_scale: float,
    texture_channels: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "image_size": image_size,
            "condition_dim": condition_dim,
            "residual_scale": residual_scale,
            "texture_channels": texture_channels,
        },
        checkpoint_path,
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("proxy_pair_manifest", type=Path)
    parser.add_argument("--scorer-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--sample-output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--include-texture-channels", action="store_true")
    parser.add_argument("--proxy-loss-weight", type=float, default=1.0)
    parser.add_argument("--masked-proxy-loss-weight", type=float, default=0.5)
    parser.add_argument("--delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--unmasked-loss-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-monotonic-fraction", type=float, default=0.8)
    parser.add_argument("--min-expected-span", type=float, default=0.5)
    parser.add_argument("--max-unmasked-delta", type=float, default=0.01)
    parser.add_argument("--mask-area-min", type=float, default=0.2)
    parser.add_argument("--mask-area-max", type=float, default=0.75)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = _load_pair_rows(args.proxy_pair_manifest)
    classes = _condition_classes(rows)
    train_rows = _split_rows(rows, "train")
    val_rows = _split_rows(rows, "val")
    test_rows = _split_rows(rows, "test")

    train_loader = DataLoader(
        ProxyPairDataset(train_rows, args.image_size, classes, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ProxyPairDataset(val_rows, args.image_size, classes, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        ProxyPairDataset(test_rows, args.image_size, classes, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
    )

    condition_dim = len(classes) * 2 + 3
    texture_channels = 6 if args.include_texture_channels else 0
    model = MaskedResidualEditor(
        condition_dim=condition_dim,
        hidden_channels=args.hidden_channels,
        residual_scale=args.residual_scale,
        texture_channels=texture_channels,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scorer, scorer_labels, scorer_image_size = _load_scorer(args.scorer_checkpoint, device)
    if scorer_image_size != args.image_size:
        raise ValueError(
            f"Scorer image size {scorer_image_size} does not match requested {args.image_size}."
        )

    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            unmasked_loss_weight=args.unmasked_loss_weight,
            proxy_loss_weight=args.proxy_loss_weight,
            masked_proxy_loss_weight=args.masked_proxy_loss_weight,
            delta_loss_weight=args.delta_loss_weight,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            unmasked_loss_weight=args.unmasked_loss_weight,
            proxy_loss_weight=args.proxy_loss_weight,
            masked_proxy_loss_weight=args.masked_proxy_loss_weight,
            delta_loss_weight=args.delta_loss_weight,
        )
        epoch_summary = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary))

        if args.checkpoint_path is not None:
            _save_checkpoint(
                model=model,
                checkpoint_path=args.checkpoint_path,
                classes=classes,
                image_size=args.image_size,
                condition_dim=condition_dim,
                residual_scale=args.residual_scale,
                texture_channels=texture_channels,
            )

    test_metrics = _run_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        device=device,
        unmasked_loss_weight=args.unmasked_loss_weight,
        proxy_loss_weight=args.proxy_loss_weight,
        masked_proxy_loss_weight=args.masked_proxy_loss_weight,
        delta_loss_weight=args.delta_loss_weight,
    )

    sample_dir = args.sample_output_dir
    val_sweep = _evaluate_cases(
        model=model,
        case_rows=_unique_case_rows(val_rows),
        image_size=args.image_size,
        classes=classes,
        scorer=scorer,
        scorer_labels=scorer_labels,
        device=device,
        output_contact_sheet=sample_dir / "val_learned_vs_proxy_contact_sheet.png"
        if sample_dir
        else None,
        split_name="val",
        min_monotonic_fraction=args.min_monotonic_fraction,
        min_expected_span=args.min_expected_span,
        max_unmasked_delta_threshold=args.max_unmasked_delta,
        mask_area_min=args.mask_area_min,
        mask_area_max=args.mask_area_max,
    )
    test_sweep = _evaluate_cases(
        model=model,
        case_rows=_unique_case_rows(test_rows),
        image_size=args.image_size,
        classes=classes,
        scorer=scorer,
        scorer_labels=scorer_labels,
        device=device,
        output_contact_sheet=sample_dir / "test_learned_vs_proxy_contact_sheet.png"
        if sample_dir
        else None,
        split_name="test",
        min_monotonic_fraction=args.min_monotonic_fraction,
        min_expected_span=args.min_expected_span,
        max_unmasked_delta_threshold=args.max_unmasked_delta,
        mask_area_min=args.mask_area_min,
        mask_area_max=args.mask_area_max,
    )

    summary = {
        "experiment": "exp03_proxy_residual_editor",
        "proxy_pair_manifest": str(args.proxy_pair_manifest),
        "scorer_checkpoint": str(args.scorer_checkpoint),
        "device": str(device),
        "image_size": args.image_size,
        "classes": classes,
        "condition_dim": condition_dim,
        "train_pair_count": len(train_rows),
        "val_pair_count": len(val_rows),
        "test_pair_count": len(test_rows),
        "train_case_count": len(_unique_case_rows(train_rows)),
        "val_case_count": len(_unique_case_rows(val_rows)),
        "test_case_count": len(_unique_case_rows(test_rows)),
        "train_dataset_distribution": dict(Counter(row["dataset_key"] for row in train_rows)),
        "val_dataset_distribution": dict(Counter(row["dataset_key"] for row in val_rows)),
        "test_dataset_distribution": dict(Counter(row["dataset_key"] for row in test_rows)),
        "model": {
            "name": "masked_residual_editor",
            "hidden_channels": args.hidden_channels,
            "residual_scale": args.residual_scale,
            "mask_constrained": True,
            "texture_channels": texture_channels,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "proxy_loss_weight": args.proxy_loss_weight,
            "masked_proxy_loss_weight": args.masked_proxy_loss_weight,
            "delta_loss_weight": args.delta_loss_weight,
            "unmasked_loss_weight": args.unmasked_loss_weight,
            "augmentation": "random_horizontal_flip",
        },
        "qa_criteria": {
            "min_monotonic_fraction": args.min_monotonic_fraction,
            "min_expected_span": args.min_expected_span,
            "max_unmasked_delta": args.max_unmasked_delta,
            "mask_area_min": args.mask_area_min,
            "mask_area_max": args.mask_area_max,
        },
        "history": history,
        "final_test_pair_metrics": test_metrics,
        "val_sweep": val_sweep,
        "test_sweep": test_sweep,
        "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else None,
        "sample_output_dir": str(sample_dir) if sample_dir else None,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
