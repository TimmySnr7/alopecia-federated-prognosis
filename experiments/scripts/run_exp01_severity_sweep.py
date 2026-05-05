"""Run a severity-conditioning sweep for a fixed image using the generative scaffold."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from PIL import Image, ImageDraw
import torch
from torchvision.transforms import Compose, InterpolationMode, Resize, ToPILImage, ToTensor

from models.diffusion.latent_diffusion import LatentDiffusionConfig, build_scaffold_model


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


def _add_label(image: Image.Image, label: str) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 88, 18), fill=(255, 255, 255))
    draw.text((4, 3), label, fill=(0, 0, 0))
    return canvas


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--noise-std", type=float, default=0.05)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = LatentDiffusionConfig(**checkpoint["model_config"])
    model = build_scaffold_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    condition_classes: list[int] = checkpoint["condition_classes"]
    transform = _transform(model_config.image_resolution)
    image = Image.open(args.image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    noisy_tensor = torch.clamp(
        image_tensor + torch.randn_like(image_tensor) * args.noise_std, 0.0, 1.0
    )

    panels = [
        _add_label(_to_pil(image_tensor[0]), "input"),
        _add_label(_to_pil(noisy_tensor[0]), "noisy"),
    ]

    with torch.no_grad():
        for target_class in condition_classes:
            one_hot = torch.zeros((1, len(condition_classes)), dtype=torch.float32, device=device)
            one_hot[0, condition_classes.index(target_class)] = 1.0
            reconstruction = model(noisy_tensor, one_hot)
            panels.append(_add_label(_to_pil(reconstruction[0]), f"target {target_class}"))

    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * width, 0))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output_path)
    print(f"Saved severity sweep to {args.output_path}")


if __name__ == "__main__":
    main()
