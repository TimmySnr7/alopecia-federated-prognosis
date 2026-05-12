"""Score an Experiment 1 severity sweep with a frozen severity classifier."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch
from torchvision.transforms import Compose, InterpolationMode, Resize, ToPILImage, ToTensor

from models.diffusion.latent_diffusion import LatentDiffusionConfig, build_scaffold_model
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


def _load_generator(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = LatentDiffusionConfig(**checkpoint["model_config"])
    model = build_scaffold_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _load_scorer(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, list[int]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
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
    return model, ordered_labels


def _expected_severity(probabilities: torch.Tensor, labels: list[int]) -> float:
    label_tensor = torch.tensor(labels, dtype=torch.float32, device=probabilities.device)
    return float((probabilities * label_tensor).sum().item())


def _monotonic_fraction(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    nondecreasing = sum(next_value >= value for value, next_value in zip(values, values[1:]))
    return nondecreasing / (len(values) - 1)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("generator_checkpoint", type=Path)
    parser.add_argument("scorer_checkpoint", type=Path)
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--panel-output-path", type=Path, required=True)
    parser.add_argument("--json-output-path", type=Path, required=True)
    parser.add_argument("--noise-std", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, generator_checkpoint = _load_generator(args.generator_checkpoint, device)
    scorer, scorer_labels = _load_scorer(args.scorer_checkpoint, device)

    model_config = LatentDiffusionConfig(**generator_checkpoint["model_config"])
    condition_classes: list[int] = generator_checkpoint["condition_classes"]
    transform = _transform(model_config.image_resolution)

    image = Image.open(args.image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    noisy_tensor = torch.clamp(
        image_tensor + torch.randn_like(image_tensor) * args.noise_std, 0.0, 1.0
    )

    panels = [
        _add_label(_to_pil(image_tensor[0]), ["input"]),
        _add_label(_to_pil(noisy_tensor[0]), ["noisy"]),
    ]
    scored_targets: list[dict[str, float | int]] = []

    with torch.no_grad():
        for target_class in condition_classes:
            one_hot = torch.zeros((1, len(condition_classes)), dtype=torch.float32, device=device)
            one_hot[0, condition_classes.index(target_class)] = 1.0
            reconstruction, _ = generator(noisy_tensor, one_hot)
            scorer_logits = scorer(reconstruction)
            probabilities = torch.softmax(scorer_logits[0], dim=0)
            predicted_index = int(probabilities.argmax().item())
            predicted_label = int(scorer_labels[predicted_index])
            confidence = float(probabilities[predicted_index].item())
            expected = _expected_severity(probabilities, scorer_labels)

            scored_targets.append(
                {
                    "target_severity": int(target_class),
                    "predicted_severity": predicted_label,
                    "predicted_confidence": confidence,
                    "expected_severity": expected,
                }
            )
            panels.append(
                _add_label(
                    _to_pil(reconstruction[0]),
                    [
                        f"target {target_class}",
                        f"pred {predicted_label} ({confidence:.2f})",
                        f"exp {expected:.2f}",
                    ],
                )
            )

    expected_values = [item["expected_severity"] for item in scored_targets]
    predicted_values = [item["predicted_severity"] for item in scored_targets]
    summary = {
        "generator_checkpoint": str(args.generator_checkpoint),
        "scorer_checkpoint": str(args.scorer_checkpoint),
        "image_path": str(args.image_path),
        "condition_classes": condition_classes,
        "scorer_labels": scorer_labels,
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
