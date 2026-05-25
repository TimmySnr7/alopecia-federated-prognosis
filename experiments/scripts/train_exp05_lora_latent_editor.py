"""Train and evaluate Experiment 05 LoRA-adapted latent editor variants."""

from __future__ import annotations

from argparse import ArgumentParser
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode, Resize, ToPILImage, ToTensor

from experiments.scripts.run_exp01_plausible_proxy_sweep import (
    _add_label,
    _build_ellipse_mask,
    _expected_severity,
    _make_proxy_edit,
    _monotonic_fraction,
    _to_pil,
)
from experiments.scripts.run_exp01_proxy_batch_eval import _load_scorer, _visual_delta_metrics
from experiments.scripts.train_exp03_proxy_residual_editor import (
    _case_passes,
    _compose_rows,
    _condition_classes,
    _load_ellipse,
    _load_pair_rows,
    _load_proxy_parameters,
    _split_rows,
    _unique_case_rows,
)


def _prompt(source_severity: int, target_severity: int, mode: str) -> str:
    delta = target_severity - source_severity
    direction = "unchanged"
    if delta > 0:
        direction = "more visible alopecia and lower hair density"
    elif delta < 0:
        direction = "higher hair density and less visible alopecia"
    if mode == "cross_attention":
        return (
            f"alopecia_source_stage_{source_severity} target_stage_{target_severity} "
            f"severity_delta_{delta}; clinical top-view scalp edit; {direction}; "
            "preserve identity, camera, lighting, background, and scalp anatomy"
        )
    if mode == "film":
        return (
            f"source severity {source_severity}; target severity {target_severity}; "
            f"ordinal change {delta}; modulate only scalp hair texture toward {direction}; "
            "no face, no new head, no background change"
        )
    if mode == "clip":
        return (
            "Edit this top-view scalp photograph so the alopecia severity looks "
            f"like stage {target_severity}, starting from stage {source_severity}. "
            "Only adjust scalp hair density and preserve the rest of the image."
        )
    raise ValueError(f"Unsupported conditioning mode: {mode}")


def _prepare_pair(
    row: dict[str, str],
    target_severity: int,
    image_size: int,
    classes: list[int],
    device: torch.device | None = None,
) -> dict[str, Any]:
    source_severity = int(float(row["source_severity"]))
    proxy_parameters = _load_proxy_parameters(row)
    ellipse = _load_ellipse(row)
    image = Image.open(row["image_path"]).convert("RGB")
    resized = Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)(image)
    blurred = resized.filter(
        ImageFilter.GaussianBlur(radius=float(proxy_parameters["texture_blur_radius"]))
    )
    target_device = device or torch.device("cpu")
    source = ToTensor()(resized).unsqueeze(0).to(target_device)
    blurred_tensor = ToTensor()(blurred).unsqueeze(0).to(target_device)
    mask = _build_ellipse_mask(
        ellipse,
        image_size=image_size,
        device=target_device,
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
        "source_pil": resized,
        "source": source[0],
        "proxy_target": proxy_target[0],
        "mask": mask[0],
        "source_severity": source_severity,
        "target_severity": target_severity,
    }


class ProxyPairLatentDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, str]],
        image_size: int,
        classes: list[int],
        conditioning_mode: str,
        augment: bool,
    ) -> None:
        self.rows = rows
        self.image_size = image_size
        self.classes = classes
        self.conditioning_mode = conditioning_mode
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        target_severity = int(float(row["target_severity"]))
        item = _prepare_pair(row, target_severity, self.image_size, self.classes)
        source = item["source"]
        proxy = item["proxy_target"]
        mask = item["mask"]
        if self.augment and random.random() < 0.5:
            source = torch.flip(source, dims=[2])
            proxy = torch.flip(proxy, dims=[2])
            mask = torch.flip(mask, dims=[2])
        return {
            "source": source,
            "proxy_target": proxy,
            "mask": mask,
            "prompt": _prompt(item["source_severity"], target_severity, self.conditioning_mode),
            "source_severity": torch.tensor(item["source_severity"], dtype=torch.long),
            "target_severity": torch.tensor(target_severity, dtype=torch.long),
            "dataset_key": row["dataset_key"],
            "case_id": row["case_id"],
        }


def _sampling_weights(rows: list[dict[str, str]]) -> list[float]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["dataset_key"]] = counts.get(row["dataset_key"], 0) + 1
    return [1.0 / counts[row["dataset_key"]] for row in rows]


def _load_pipe(model_id: str, device: torch.device, dtype: torch.dtype):
    from diffusers import StableDiffusionInstructPix2PixPipeline
    from diffusers.schedulers import DDPMScheduler, EulerAncestralDiscreteScheduler

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    for module in [pipe.vae, pipe.text_encoder, pipe.unet]:
        module.requires_grad_(False)
    return pipe, EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


def _attach_lora(unet, rank: int, alpha: int) -> None:
    from peft import LoraConfig

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(config)
    for parameter in unet.parameters():
        parameter.requires_grad_(False)
    for name, parameter in unet.named_parameters():
        if "lora" in name.lower():
            parameter.requires_grad_(True)


def _encode_prompt(pipe, prompts: list[str], device: torch.device) -> torch.Tensor:
    text_inputs = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = (
        text_inputs.attention_mask.to(device)
        if getattr(pipe.text_encoder.config, "use_attention_mask", False)
        else None
    )
    return pipe.text_encoder(input_ids, attention_mask=attention_mask)[0]


def _preprocess_for_vae(images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (images.to(dtype=dtype) * 2.0 - 1.0).clamp(-1.0, 1.0)


def _train(
    pipe,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    epochs: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    mask_loss_weight: float,
) -> list[dict[str, float]]:
    pipe.unet.train()
    params = [parameter for parameter in pipe.unet.parameters() if parameter.requires_grad]
    optimizer = AdamW(params, lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    scaling_factor = pipe.vae.config.scaling_factor

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_examples = 0
        optimizer.zero_grad()
        for step, batch in enumerate(loader, start=1):
            source = batch["source"].to(device)
            proxy = batch["proxy_target"].to(device)
            mask = batch["mask"].to(device)
            prompts = list(batch["prompt"])
            with torch.no_grad():
                target_latents = (
                    pipe.vae.encode(_preprocess_for_vae(proxy, dtype)).latent_dist.sample()
                    * scaling_factor
                )
                source_latents = (
                    pipe.vae.encode(_preprocess_for_vae(source, dtype)).latent_dist.mode()
                    * scaling_factor
                )
                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(
                    0,
                    pipe.scheduler.config.num_train_timesteps,
                    (target_latents.shape[0],),
                    device=device,
                    dtype=torch.long,
                )
                noisy_latents = pipe.scheduler.add_noise(target_latents, noise, timesteps)
                encoder_hidden_states = _encode_prompt(pipe, prompts, device)

            latent_input = torch.cat([noisy_latents, source_latents], dim=1)
            model_pred = pipe.unet(latent_input, timesteps, encoder_hidden_states).sample
            latent_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none").mean(
                dim=(1, 2, 3)
            )
            with torch.no_grad():
                resized_mask = F.interpolate(mask, size=target_latents.shape[-2:], mode="area")
                mask_weight = 1.0 + mask_loss_weight * resized_mask.flatten(start_dim=1).mean(dim=1)
            loss = (latent_loss * mask_weight).mean() / gradient_accumulation_steps
            loss.backward()
            if step % gradient_accumulation_steps == 0 or step == len(loader):
                optimizer.step()
                optimizer.zero_grad()
            batch_size = source.shape[0]
            total_loss += float(loss.item()) * gradient_accumulation_steps * batch_size
            total_examples += batch_size
        epoch_summary = {"epoch": epoch, "train_loss": total_loss / max(total_examples, 1)}
        history.append(epoch_summary)
        print(json.dumps(epoch_summary))
    pipe.unet.eval()
    return history


def _score_image(scorer: torch.nn.Module, labels: list[int], image: torch.Tensor) -> dict[str, float | int]:
    logits = scorer(image)
    probabilities = torch.softmax(logits[0], dim=0)
    predicted_index = int(probabilities.argmax().item())
    return {
        "predicted_severity": int(labels[predicted_index]),
        "predicted_confidence": float(probabilities[predicted_index].item()),
        "expected_severity": _expected_severity(probabilities, labels),
    }


def _frechet_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    import numpy as np
    from scipy.linalg import sqrtm

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    mu_x, mu_y = x_np.mean(axis=0), y_np.mean(axis=0)
    cov_x, cov_y = np.cov(x_np, rowvar=False), np.cov(y_np, rowvar=False)
    covmean = sqrtm(cov_x @ cov_y)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu_x - mu_y) @ (mu_x - mu_y)) + np.trace(cov_x + cov_y - 2 * covmean))


def _feature_model(device: torch.device):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _evaluate(
    pipe,
    generation_scheduler,
    scorer: torch.nn.Module,
    scorer_labels: list[int],
    rows: list[dict[str, str]],
    image_size: int,
    classes: list[int],
    device: torch.device,
    dtype: torch.dtype,
    output_contact_sheet: Path | None,
    split_name: str,
    conditioning_mode: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    image_guidance_scale: float,
    mask_blend_strength: float,
    min_monotonic_fraction: float,
    min_expected_span: float,
    max_unmasked_delta_threshold: float,
    mask_area_min: float,
    mask_area_max: float,
) -> dict[str, Any]:
    import lpips

    pipe.scheduler = generation_scheduler
    pipe.unet.eval()
    scorer.eval()
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    features = _feature_model(device)
    contact_rows: list[list[Image.Image]] = []
    case_summaries: list[dict[str, Any]] = []
    latent_feature_values: list[torch.Tensor] = []
    proxy_feature_values: list[torch.Tensor] = []

    with torch.no_grad():
        for case_index, row in enumerate(rows):
            source_severity = int(float(row["source_severity"]))
            source_item = _prepare_pair(row, source_severity, image_size, classes, device=device)
            source = source_item["source"].unsqueeze(0)
            mask = source_item["mask"].unsqueeze(0)
            source_pil = ToPILImage()(source_item["source"].cpu())
            source_score = _score_image(scorer, scorer_labels, source)
            latent_expected: list[float] = []
            proxy_expected: list[float] = []
            latent_predicted: list[int] = []
            proxy_predicted: list[int] = []
            latent_l1_values: list[float] = []
            latent_lpips_values: list[float] = []
            proxy_lpips_values: list[float] = []
            texture_guard_values: list[float] = []
            max_unmasked = 0.0
            max_masked = 0.0
            proxy_panels = [
                _add_label(
                    _to_pil(source[0]),
                    [f"{split_name} source", f"sev {source_severity}", row["dataset_key"][:24]],
                )
            ]
            latent_panels = [
                _add_label(_to_pil(source[0]), ["lora", f"source exp {source_score['expected_severity']:.2f}"])
            ]
            for target_index, target_severity in enumerate(classes):
                item = _prepare_pair(row, target_severity, image_size, classes, device=device)
                proxy = item["proxy_target"].unsqueeze(0)
                if target_severity == source_severity:
                    latent = source.clone()
                else:
                    generator = torch.Generator(device=device).manual_seed(
                        seed + case_index * 100 + target_index
                    )
                    image = pipe(
                        prompt=_prompt(source_severity, target_severity, conditioning_mode),
                        image=source_pil,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        image_guidance_scale=image_guidance_scale,
                        generator=generator,
                    ).images[0].convert("RGB")
                    edited = ToTensor()(image).unsqueeze(0).to(device)
                    if edited.shape[-2:] != source.shape[-2:]:
                        edited = F.interpolate(edited, size=source.shape[-2:], mode="bilinear", align_corners=False)
                    blend = (mask * mask_blend_strength).clamp(0.0, 1.0)
                    latent = (source * (1.0 - blend) + edited * blend).clamp(0.0, 1.0)

                latent_score = _score_image(scorer, scorer_labels, latent)
                proxy_score = _score_image(scorer, scorer_labels, proxy)
                latent_expected.append(float(latent_score["expected_severity"]))
                proxy_expected.append(float(proxy_score["expected_severity"]))
                latent_predicted.append(int(latent_score["predicted_severity"]))
                proxy_predicted.append(int(proxy_score["predicted_severity"]))
                metrics = _visual_delta_metrics(source, latent, mask)
                max_unmasked = max(max_unmasked, metrics["unmasked_mean_absolute_delta"])
                max_masked = max(max_masked, metrics["masked_mean_absolute_delta"])
                latent_l1_values.append(float((latent - proxy).abs().mean().item()))
                latent_lpips = float(lpips_model(latent * 2 - 1, proxy * 2 - 1).item())
                proxy_lpips = float(lpips_model(proxy * 2 - 1, source * 2 - 1).item())
                latent_lpips_values.append(latent_lpips)
                proxy_lpips_values.append(proxy_lpips)
                texture_guard_values.append(float(((latent - proxy).abs() * mask).sum().item() / (mask.sum().item() * 3 + 1e-8)))
                latent_feature_values.append(features(latent).float().cpu())
                proxy_feature_values.append(features(proxy).float().cpu())
                proxy_panels.append(
                    _add_label(
                        _to_pil(proxy[0]),
                        [f"proxy t{target_severity}", f"pred {proxy_score['predicted_severity']}", f"exp {proxy_score['expected_severity']:.2f}"],
                    )
                )
                latent_panels.append(
                    _add_label(
                        _to_pil(latent[0]),
                        [f"lora t{target_severity}", f"pred {latent_score['predicted_severity']}", f"exp {latent_score['expected_severity']:.2f}"],
                    )
                )

            latent_monotonic = _monotonic_fraction(latent_expected)
            proxy_monotonic = _monotonic_fraction(proxy_expected)
            latent_span = max(latent_expected) - min(latent_expected)
            proxy_span = max(proxy_expected) - min(proxy_expected)
            mask_area_fraction = float(mask.mean().item())
            latent_auto_pass = _case_passes(
                latent_monotonic,
                latent_span,
                max_unmasked,
                mask_area_fraction,
                min_monotonic_fraction,
                min_expected_span,
                max_unmasked_delta_threshold,
                mask_area_min,
                mask_area_max,
            )
            proxy_auto_pass = _case_passes(
                proxy_monotonic,
                proxy_span,
                0.0,
                mask_area_fraction,
                min_monotonic_fraction,
                min_expected_span,
                max_unmasked_delta_threshold,
                mask_area_min,
                mask_area_max,
            )
            case_summaries.append(
                {
                    "case_id": row["case_id"],
                    "dataset_key": row["dataset_key"],
                    "split": split_name,
                    "source_severity": source_severity,
                    "proxy_expected_severities": proxy_expected,
                    "latent_expected_severities": latent_expected,
                    "proxy_predicted_severities": proxy_predicted,
                    "latent_predicted_severities": latent_predicted,
                    "proxy_monotonic_fraction": proxy_monotonic,
                    "latent_monotonic_fraction": latent_monotonic,
                    "proxy_expected_span": proxy_span,
                    "latent_expected_span": latent_span,
                    "latent_mean_l1_to_proxy": mean(latent_l1_values),
                    "latent_mean_lpips_to_proxy": mean(latent_lpips_values),
                    "proxy_mean_lpips_to_source": mean(proxy_lpips_values),
                    "anatomical_guard_masked_l1_to_proxy": mean(texture_guard_values),
                    "latent_max_masked_mean_absolute_delta": max_masked,
                    "latent_max_unmasked_mean_absolute_delta": max_unmasked,
                    "mask_area_fraction": mask_area_fraction,
                    "proxy_auto_pass": proxy_auto_pass,
                    "latent_auto_pass": latent_auto_pass,
                }
            )
            contact_rows.extend([proxy_panels, latent_panels])

    if output_contact_sheet is not None and contact_rows:
        output_contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        _compose_rows(contact_rows).save(output_contact_sheet)

    latent_pass_count = sum(case["latent_auto_pass"] for case in case_summaries)
    proxy_pass_count = sum(case["proxy_auto_pass"] for case in case_summaries)
    latent_features = torch.cat(latent_feature_values, dim=0)
    proxy_features = torch.cat(proxy_feature_values, dim=0)
    by_dataset: dict[str, dict[str, float]] = {}
    for dataset_key in sorted({case["dataset_key"] for case in case_summaries}):
        subset = [case for case in case_summaries if case["dataset_key"] == dataset_key]
        by_dataset[dataset_key] = {
            "case_count": len(subset),
            "mean_latent_expected_span": mean([case["latent_expected_span"] for case in subset]),
            "mean_latent_monotonic_fraction": mean([case["latent_monotonic_fraction"] for case in subset]),
            "latent_auto_pass_rate": sum(case["latent_auto_pass"] for case in subset) / max(len(subset), 1),
        }
    return {
        "split": split_name,
        "case_count": len(case_summaries),
        "latent_auto_pass_count": latent_pass_count,
        "latent_auto_pass_rate": latent_pass_count / max(len(case_summaries), 1),
        "proxy_auto_pass_count": proxy_pass_count,
        "proxy_auto_pass_rate": proxy_pass_count / max(len(case_summaries), 1),
        "mean_latent_monotonic_fraction": mean([case["latent_monotonic_fraction"] for case in case_summaries]),
        "mean_proxy_monotonic_fraction": mean([case["proxy_monotonic_fraction"] for case in case_summaries]),
        "mean_latent_expected_span": mean([case["latent_expected_span"] for case in case_summaries]),
        "mean_proxy_expected_span": mean([case["proxy_expected_span"] for case in case_summaries]),
        "mean_latent_l1_to_proxy": mean([case["latent_mean_l1_to_proxy"] for case in case_summaries]),
        "mean_latent_lpips_to_proxy": mean([case["latent_mean_lpips_to_proxy"] for case in case_summaries]),
        "mean_proxy_lpips_to_source": mean([case["proxy_mean_lpips_to_source"] for case in case_summaries]),
        "domain_fid_resnet18_proxy_reference": _frechet_distance(latent_features, proxy_features),
        "mean_anatomical_guard_masked_l1_to_proxy": mean([case["anatomical_guard_masked_l1_to_proxy"] for case in case_summaries]),
        "mean_latent_max_unmasked_delta": mean([case["latent_max_unmasked_mean_absolute_delta"] for case in case_summaries]),
        "by_dataset": by_dataset,
        "cases": case_summaries,
        "contact_sheet": str(output_contact_sheet) if output_contact_sheet else None,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("proxy_pair_manifest", type=Path)
    parser.add_argument("--scorer-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--lora-output-dir", type=Path, required=True)
    parser.add_argument("--sample-output-dir", type=Path, default=None)
    parser.add_argument("--model-id", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--conditioning-mode", choices=["cross_attention", "film", "clip"], default="cross_attention")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=4)
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--image-guidance-scale", type=float, default=2.0)
    parser.add_argument("--mask-blend-strength", type=float, default=0.5)
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
    train_dataset = ProxyPairLatentDataset(train_rows, args.image_size, classes, args.conditioning_mode, augment=True)
    sampler = WeightedRandomSampler(torch.tensor(_sampling_weights(train_rows), dtype=torch.double), len(train_rows), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    # Keep training/evaluation in fp32 for this tiny medical-image adaptation;
    # the first fp16 smoke run was numerically unstable on AI5.
    dtype = torch.float32
    pipe, generation_scheduler = _load_pipe(args.model_id, device, dtype)
    _attach_lora(pipe.unet, rank=args.rank, alpha=args.alpha)
    trainable_params = sum(parameter.numel() for parameter in pipe.unet.parameters() if parameter.requires_grad)
    history = _train(
        pipe=pipe,
        loader=train_loader,
        device=device,
        dtype=dtype,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mask_loss_weight=args.mask_loss_weight,
    )
    args.lora_output_dir.mkdir(parents=True, exist_ok=True)
    lora_state = {
        key: value.detach().cpu()
        for key, value in pipe.unet.state_dict().items()
        if "lora" in key.lower()
    }
    torch.save(lora_state, args.lora_output_dir / "unet_lora_state_dict.pt")
    (args.lora_output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "conditioning_mode": args.conditioning_mode,
                "rank": args.rank,
                "alpha": args.alpha,
                "format": "filtered_unet_state_dict_lora_keys",
            },
            indent=2,
        )
        + "\n"
    )

    scorer, scorer_labels, scorer_image_size = _load_scorer(args.scorer_checkpoint, device)
    if scorer_image_size != args.image_size:
        raise ValueError(f"Scorer image size {scorer_image_size} != requested {args.image_size}")
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    sample_dir = args.sample_output_dir
    val_sweep = _evaluate(
        pipe, generation_scheduler, scorer, scorer_labels, _unique_case_rows(val_rows),
        args.image_size, classes, device, dtype,
        sample_dir / "val_lora_vs_proxy_contact_sheet.png" if sample_dir else None,
        "val", args.conditioning_mode, args.seed, args.num_inference_steps,
        args.guidance_scale, args.image_guidance_scale, args.mask_blend_strength,
        args.min_monotonic_fraction, args.min_expected_span, args.max_unmasked_delta,
        args.mask_area_min, args.mask_area_max,
    )
    test_sweep = _evaluate(
        pipe, generation_scheduler, scorer, scorer_labels, _unique_case_rows(test_rows),
        args.image_size, classes, device, dtype,
        sample_dir / "test_lora_vs_proxy_contact_sheet.png" if sample_dir else None,
        "test", args.conditioning_mode, args.seed + 10_000, args.num_inference_steps,
        args.guidance_scale, args.image_guidance_scale, args.mask_blend_strength,
        args.min_monotonic_fraction, args.min_expected_span, args.max_unmasked_delta,
        args.mask_area_min, args.mask_area_max,
    )
    summary = {
        "experiment": "exp05_lora_latent_editor",
        "model_id": args.model_id,
        "conditioning_mode": args.conditioning_mode,
        "proxy_pair_manifest": str(args.proxy_pair_manifest),
        "scorer_checkpoint": str(args.scorer_checkpoint),
        "device": str(device),
        "dtype": str(dtype),
        "classes": classes,
        "image_size": args.image_size,
        "train_pair_count": len(train_rows),
        "val_case_count": len(_unique_case_rows(val_rows)),
        "test_case_count": len(_unique_case_rows(test_rows)),
        "trainable_lora_parameters": trainable_params,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "rank": args.rank,
            "alpha": args.alpha,
            "mask_loss_weight": args.mask_loss_weight,
            "augmentation": "horizontal_flip",
            "history": history,
        },
        "generation": {
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "image_guidance_scale": args.image_guidance_scale,
            "mask_blend_strength": args.mask_blend_strength,
            "seed": args.seed,
        },
        "qa_criteria": {
            "min_monotonic_fraction": args.min_monotonic_fraction,
            "min_expected_span": args.min_expected_span,
            "max_unmasked_delta": args.max_unmasked_delta,
            "mask_area_min": args.mask_area_min,
            "mask_area_max": args.mask_area_max,
        },
        "val_sweep": val_sweep,
        "test_sweep": test_sweep,
        "lora_output_dir": str(args.lora_output_dir),
        "sample_output_dir": str(sample_dir) if sample_dir else None,
        "limitations": [
            "Executable v1 uses InstructPix2Pix as the image-conditioned latent backbone because the requested SD2.1 image-conditioned checkpoint was not available on AI5.",
            "FiLM mode is represented as a conditioning-text ablation, not a custom FiLM-modified UNet.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
