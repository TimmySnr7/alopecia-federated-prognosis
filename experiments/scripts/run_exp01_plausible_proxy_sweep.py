"""Run a morphology-inspired visual plausibility proxy sweep for Experiment 1."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
import torch
from torch import nn
from torchvision.transforms import InterpolationMode, Resize, ToPILImage, ToTensor

from models.severity_grading.norwood_classifier import (
    NorwoodClassifierConfig,
    build_baseline_classifier,
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


def _build_ellipse_mask(
    ellipse: list[float],
    image_size: int,
    device: torch.device,
    blur_radius: float,
) -> torch.Tensor:
    cx, cy, rx, ry = ellipse
    mask_image = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(mask_image)
    draw.ellipse(
        (
            int((cx - rx) * image_size),
            int((cy - ry) * image_size),
            int((cx + rx) * image_size),
            int((cy + ry) * image_size),
        ),
        fill=255,
    )
    if blur_radius > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return ToTensor()(mask_image).unsqueeze(0).to(device).clamp(0.0, 1.0)


def _visual_delta_metrics(
    source_image: torch.Tensor,
    edited_image: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    delta = (edited_image - source_image).abs()
    inverse_mask = 1.0 - mask
    return {
        "mean_absolute_delta": float(delta.mean().item()),
        "max_absolute_delta": float(delta.max().item()),
        "mask_area_fraction": float(mask.mean().item()),
        "masked_mean_absolute_delta": float(
            (delta * mask).sum().item() / (mask.sum().item() * delta.size(1) + 1e-8)
        ),
        "unmasked_mean_absolute_delta": float(
            (delta * inverse_mask).sum().item()
            / (inverse_mask.sum().item() * delta.size(1) + 1e-8)
        ),
    }


def _make_proxy_edit(
    source: torch.Tensor,
    blurred: torch.Tensor,
    mask: torch.Tensor,
    target_severity: int,
    source_severity: float,
    min_severity: int,
    max_severity: int,
    hair_enhancement_strength: float,
    hair_suppression_strength: float,
    smoothing_strength: float,
    tone_shift_strength: float,
) -> torch.Tensor:
    dark_detail = (blurred - source).clamp(min=0.0)
    edited = source.clone()

    if target_severity < source_severity:
        alpha = (source_severity - target_severity) / max(source_severity - min_severity, 1e-8)
        # Lower severity proxy: emphasize existing short-hair/stubble texture.
        edited = edited - alpha * hair_enhancement_strength * dark_detail
        edited = edited - alpha * tone_shift_strength * 0.5
    elif target_severity > source_severity:
        alpha = (target_severity - source_severity) / max(max_severity - source_severity, 1e-8)
        # Higher severity proxy: suppress dark stubble and smooth toward local scalp tone.
        edited = edited + alpha * hair_suppression_strength * dark_detail
        edited = edited + alpha * smoothing_strength * (blurred - edited)
        edited = edited + alpha * tone_shift_strength

    edited = torch.clamp(edited, 0.0, 1.0)
    return torch.clamp(source * (1.0 - mask) + edited * mask, 0.0, 1.0)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("scorer_checkpoint", type=Path)
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--panel-output-path", type=Path, required=True)
    parser.add_argument("--json-output-path", type=Path, required=True)
    parser.add_argument("--source-severity", type=float, required=True)
    parser.add_argument(
        "--ellipse-mask",
        nargs=4,
        type=float,
        metavar=("CX", "CY", "RX", "RY"),
        required=True,
        help="Normalized ellipse mask in resized-image coordinates.",
    )
    parser.add_argument("--mask-blur-radius", type=float, default=8.0)
    parser.add_argument("--texture-blur-radius", type=float, default=3.0)
    parser.add_argument("--hair-enhancement-strength", type=float, default=1.2)
    parser.add_argument("--hair-suppression-strength", type=float, default=1.5)
    parser.add_argument("--smoothing-strength", type=float, default=0.35)
    parser.add_argument("--tone-shift-strength", type=float, default=0.03)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer, labels, image_size = _load_scorer(args.scorer_checkpoint, device)
    resize = Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)

    image = Image.open(args.image_path).convert("RGB")
    resized_image = resize(image)
    blurred_image = resized_image.filter(ImageFilter.GaussianBlur(radius=args.texture_blur_radius))
    source = ToTensor()(resized_image).unsqueeze(0).to(device)
    blurred = ToTensor()(blurred_image).unsqueeze(0).to(device)
    mask = _build_ellipse_mask(
        args.ellipse_mask,
        image_size=image_size,
        device=device,
        blur_radius=args.mask_blur_radius,
    )

    panels = [_add_label(_to_pil(source[0]), ["input"])]
    scored_targets: list[dict[str, float | int]] = []
    min_severity = min(labels)
    max_severity = max(labels)

    with torch.no_grad():
        for target_severity in labels:
            edited = _make_proxy_edit(
                source=source,
                blurred=blurred,
                mask=mask,
                target_severity=target_severity,
                source_severity=args.source_severity,
                min_severity=min_severity,
                max_severity=max_severity,
                hair_enhancement_strength=args.hair_enhancement_strength,
                hair_suppression_strength=args.hair_suppression_strength,
                smoothing_strength=args.smoothing_strength,
                tone_shift_strength=args.tone_shift_strength,
            )
            logits = scorer(edited)
            probabilities = torch.softmax(logits[0], dim=0)
            predicted_index = int(probabilities.argmax().item())
            predicted_label = int(labels[predicted_index])
            confidence = float(probabilities[predicted_index].item())
            expected = _expected_severity(probabilities, labels)
            summary = {
                "target_severity": int(target_severity),
                "predicted_severity": predicted_label,
                "predicted_confidence": confidence,
                "expected_severity": expected,
                **_visual_delta_metrics(source, edited, mask),
            }
            scored_targets.append(summary)
            panels.append(
                _add_label(
                    _to_pil(edited[0]),
                    [
                        f"target {target_severity}",
                        f"pred {predicted_label} ({confidence:.2f})",
                        f"exp {expected:.2f}",
                    ],
                )
            )

    expected_values = [item["expected_severity"] for item in scored_targets]
    predicted_values = [item["predicted_severity"] for item in scored_targets]
    summary = {
        "scorer_checkpoint": str(args.scorer_checkpoint),
        "image_path": str(args.image_path),
        "source_severity": args.source_severity,
        "ellipse_mask": args.ellipse_mask,
        "mask_blur_radius": args.mask_blur_radius,
        "texture_blur_radius": args.texture_blur_radius,
        "hair_enhancement_strength": args.hair_enhancement_strength,
        "hair_suppression_strength": args.hair_suppression_strength,
        "smoothing_strength": args.smoothing_strength,
        "tone_shift_strength": args.tone_shift_strength,
        "targets": scored_targets,
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
