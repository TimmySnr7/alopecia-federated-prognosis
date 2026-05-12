"""Run an explicit scorer-guided residual-edit sweep for Experiment 1."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
import torch
from torch import nn
import torch.nn.functional as F
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


def _load_mask(
    mask_path: Path,
    image_size: int,
    device: torch.device,
    blur_radius: float,
    threshold: float,
) -> torch.Tensor:
    mask_image = Image.open(mask_path).convert("L")
    mask_image = mask_image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    mask_tensor = ToTensor()(mask_image).unsqueeze(0).to(device)
    mask_tensor = (mask_tensor >= threshold).float()

    if blur_radius > 0:
        pil_mask = _to_pil(mask_tensor[0].repeat(3, 1, 1)).convert("L")
        pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        mask_tensor = ToTensor()(pil_mask).unsqueeze(0).to(device)
    return mask_tensor.clamp(0.0, 1.0)


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


def _channel_balance(tensor: torch.Tensor) -> torch.Tensor:
    channel_mean = tensor.mean(dim=1, keepdim=True)
    return (tensor - channel_mean).abs().mean()


def _optimize_target(
    scorer: nn.Module,
    source_image: torch.Tensor,
    target_index: int,
    labels: list[int],
    steps: int,
    learning_rate: float,
    residual_scale: float,
    residual_resolution: int,
    mask: torch.Tensor | None,
    l2_weight: float,
    tv_weight: float,
    color_balance_weight: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    image_height, image_width = source_image.shape[-2:]
    residual_shape = (
        1,
        3,
        residual_resolution,
        residual_resolution,
    )
    residual_parameter = torch.zeros(
        residual_shape,
        dtype=source_image.dtype,
        device=source_image.device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([residual_parameter], lr=learning_rate)
    target_tensor = torch.tensor([target_index], dtype=torch.long, device=source_image.device)

    final_image = source_image.detach()
    final_logits = None
    for _ in range(steps):
        optimizer.zero_grad()
        residual = F.interpolate(
            residual_parameter,
            size=(image_height, image_width),
            mode="bilinear",
            align_corners=False,
        )
        if mask is not None:
            residual = residual * mask
        bounded_residual = torch.tanh(residual)
        edited = torch.clamp(source_image + residual_scale * bounded_residual, 0.0, 1.0)
        logits = scorer(edited)
        target_loss = nn.functional.cross_entropy(logits, target_tensor)
        l2_loss = residual.pow(2).mean()
        tv_loss = _total_variation(residual)
        color_balance_loss = _channel_balance(residual)
        loss = (
            target_loss
            + l2_weight * l2_loss
            + tv_weight * tv_loss
            + color_balance_weight * color_balance_loss
        )
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
    parser.add_argument("--mask-path", type=Path, default=None)
    parser.add_argument("--mask-blur-radius", type=float, default=3.0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--residual-scale", type=float, default=0.15)
    parser.add_argument("--residual-resolution", type=int, default=64)
    parser.add_argument("--l2-weight", type=float, default=0.01)
    parser.add_argument("--tv-weight", type=float, default=0.02)
    parser.add_argument("--color-balance-weight", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer, labels, image_size = _load_scorer(args.scorer_checkpoint, device)
    transform = _transform(image_size)

    image = Image.open(args.image_path).convert("RGB")
    source_tensor = transform(image).unsqueeze(0).to(device)
    mask = (
        _load_mask(
            args.mask_path,
            image_size=image_size,
            device=device,
            blur_radius=args.mask_blur_radius,
            threshold=args.mask_threshold,
        )
        if args.mask_path is not None
        else None
    )
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
            residual_resolution=args.residual_resolution,
            mask=mask,
            l2_weight=args.l2_weight,
            tv_weight=args.tv_weight,
            color_balance_weight=args.color_balance_weight,
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
        "mask_path": str(args.mask_path) if args.mask_path else None,
        "mask_blur_radius": args.mask_blur_radius,
        "mask_threshold": args.mask_threshold,
        "steps": args.steps,
        "learning_rate": args.lr,
        "residual_scale": args.residual_scale,
        "residual_resolution": args.residual_resolution,
        "l2_weight": args.l2_weight,
        "tv_weight": args.tv_weight,
        "color_balance_weight": args.color_balance_weight,
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
