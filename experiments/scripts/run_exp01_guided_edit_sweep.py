"""Run an explicit scorer-guided residual-edit sweep for Experiment 1."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch
from torch import nn
from torchvision.transforms import Compose, InterpolationMode, Resize, ToPILImage, ToTensor

from models.severity_grading.norwood_classifier import (
    NorwoodClassifierConfig,
    build_baseline_classifier,
)


def _transform(image_size: int) -> Compose:
    return Compose(
        [
            Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            ToTensor(),
        ]
    )


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    image = ToPILImage()(tensor.cpu().clamp(0.0, 1.0))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _add_label(image: Image.Image, lines: list[str]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 176, 46), fill=(255, 255, 255))
    for index, line in enumerate(lines):
        draw.text((4, 3 + 14 * index), line, fill=(0, 0, 0))
    return canvas


def _load_scorer(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, list[int], int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    image_size = int(checkpoint.get("image_size", 256))
    model = build_baseline_classifier(
        NorwoodClassifierConfig(
            backbone=config["backbone"],
            class_count=config["class_count"],
            pretrained=False,
        )
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    label_map = {int(label): int(index) for label, index in checkpoint["label_map"].items()}
    ordered_labels = [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]
    return model, ordered_labels, image_size


def _expected_severity(probabilities: torch.Tensor, labels: list[int]) -> float:
    label_tensor = torch.tensor(labels, dtype=torch.float32, device=probabilities.device)
    return float((probabilities * label_tensor).sum().item())


def _monotonic_fraction(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    nondecreasing = sum(next_value >= value for value, next_value in zip(values, values[1:]))
    return nondecreasing / (len(values) - 1)


def _total_variation(tensor: torch.Tensor) -> torch.Tensor:
    x_diff = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    y_diff = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
    return x_diff.abs().mean() + y_diff.abs().mean()


def _optimize_target(
    scorer: nn.Module,
    source_image: torch.Tensor,
    target_index: int,
    labels: list[int],
    steps: int,
    learning_rate: float,
    residual_scale: float,
    l2_weight: float,
    tv_weight: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    residual = torch.zeros_like(source_image, requires_grad=True)
    optimizer = torch.optim.Adam([residual], lr=learning_rate)
    target_tensor = torch.tensor([target_index], dtype=torch.long, device=source_image.device)

    final_image = source_image.detach()
    final_logits = None
    for _ in range(steps):
        optimizer.zero_grad()
        edited = torch.clamp(source_image + residual_scale * torch.tanh(residual), 0.0, 1.0)
        logits = scorer(edited)
        target_loss = nn.functional.cross_entropy(logits, target_tensor)
        l2_loss = residual.pow(2).mean()
        tv_loss = _total_variation(residual)
        loss = target_loss + l2_weight * l2_loss + tv_weight * tv_loss
        loss.backward()
        optimizer.step()
        final_image = edited.detach()
        final_logits = logits.detach()

    probabilities = torch.softmax(final_logits[0], dim=0)
    predicted_index = int(probabilities.argmax().item())
    predicted_label = int(labels[predicted_index])
    confidence = float(probabilities[predicted_index].item())
    expected = _expected_severity(probabilities, labels)
    return final_image, {
        "predicted_severity": predicted_label,
        "predicted_confidence": confidence,
        "expected_severity": expected,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("scorer_checkpoint", type=Path)
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--panel-output-path", type=Path, required=True)
    parser.add_argument("--json-output-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--residual-scale", type=float, default=0.15)
    parser.add_argument("--l2-weight", type=float, default=0.01)
    parser.add_argument("--tv-weight", type=float, default=0.02)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer, labels, image_size = _load_scorer(args.scorer_checkpoint, device)
    transform = _transform(image_size)

    image = Image.open(args.image_path).convert("RGB")
    source_tensor = transform(image).unsqueeze(0).to(device)
    panels = [_add_label(_to_pil(source_tensor[0]), ["input"])]

    target_summaries: list[dict[str, float | int]] = []
    for target_index, target_label in enumerate(labels):
        edited, stats = _optimize_target(
            scorer=scorer,
            source_image=source_tensor,
            target_index=target_index,
            labels=labels,
            steps=args.steps,
            learning_rate=args.lr,
            residual_scale=args.residual_scale,
            l2_weight=args.l2_weight,
            tv_weight=args.tv_weight,
        )
        summary = {
            "target_severity": int(target_label),
            **stats,
        }
        target_summaries.append(summary)
        panels.append(
            _add_label(
                _to_pil(edited[0]),
                [
                    f"target {target_label}",
                    f"pred {summary['predicted_severity']} ({summary['predicted_confidence']:.2f})",
                    f"exp {summary['expected_severity']:.2f}",
                ],
            )
        )

    expected_values = [item["expected_severity"] for item in target_summaries]
    predicted_values = [item["predicted_severity"] for item in target_summaries]
    summary = {
        "scorer_checkpoint": str(args.scorer_checkpoint),
        "image_path": str(args.image_path),
        "steps": args.steps,
        "learning_rate": args.lr,
        "residual_scale": args.residual_scale,
        "l2_weight": args.l2_weight,
        "tv_weight": args.tv_weight,
        "targets": target_summaries,
        "expected_severity_monotonic_fraction": _monotonic_fraction(expected_values),
        "predicted_severity_monotonic_fraction": _monotonic_fraction(predicted_values),
    }

    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * width, 0))

    args.panel_output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.panel_output_path)
    args.json_output_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
