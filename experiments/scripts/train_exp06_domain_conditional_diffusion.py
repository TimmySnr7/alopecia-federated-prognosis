"""Train Experiment 06 domain-specific conditional diffusion variants.

This script is intentionally stricter than the Experiment 05 LoRA editor. It
uses the Experiment 02 QA-gated proxy-pair manifest as the data contract,
generates proxy targets on the fly, injects explicit severity-delta conditioning
as an extra cross-attention token, and supports ControlNet-style spatial
conditioning plus an optional edited-region anatomical guard discriminator.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageFilter, ImageOps
import torch
import torch.nn.functional as F
from torch import nn
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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _severity_prompt(source_severity: int, target_severity: int) -> str:
    delta = target_severity - source_severity
    direction = "unchanged scalp hair density"
    if delta > 0:
        direction = "less scalp hair coverage and more visible alopecia"
    elif delta < 0:
        direction = "more scalp hair coverage and less visible alopecia"
    return (
        "clinical top-view scalp photograph, preserve camera, pose, identity, "
        f"and background; source severity {source_severity}; target severity "
        f"{target_severity}; severity delta {delta}; {direction}; no face, "
        "no new head, no non-scalp content"
    )


def _prepare_proxy_pair(
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
        "source": source[0],
        "proxy_target": proxy_target[0],
        "mask": mask[0],
        "source_pil": resized,
        "source_severity": source_severity,
        "target_severity": target_severity,
    }


def _soft_edge_hint(source: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Build a source-structure hint without introducing an OpenCV dependency."""
    panels: list[torch.Tensor] = []
    to_pil = ToPILImage()
    to_tensor = ToTensor()
    for image, mask_image in zip(source, mask):
        gray = ImageOps.grayscale(to_pil(image.cpu()))
        edges = gray.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=1.25))
        edge_tensor = to_tensor(edges).to(image.device, dtype=image.dtype)
        edge_tensor = (edge_tensor * mask_image + image.mean(dim=0, keepdim=True) * (1.0 - mask_image)).clamp(0.0, 1.0)
        panels.append(edge_tensor.repeat(3, 1, 1))
    controlnet_hint = torch.stack(panels, dim=0)
    if controlnet_hint.shape[1] == 1:
        controlnet_hint = controlnet_hint.expand(-1, 3, -1, -1)
    if controlnet_hint.shape[1] != 3:
        raise ValueError(f"ControlNet hint must be 3-channel, got {tuple(controlnet_hint.shape)}")
    return controlnet_hint


def _check_proxy_determinism(
    rows: list[dict[str, str]],
    image_size: int,
    classes: list[int],
    sample_count: int = 3,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for row in rows[:sample_count]:
        target_severity = int(float(row["target_severity"]))
        first = _prepare_proxy_pair(row, target_severity, image_size, classes)
        second = _prepare_proxy_pair(row, target_severity, image_size, classes)
        source_delta = float((first["source"] - second["source"]).abs().max().item())
        proxy_delta = float((first["proxy_target"] - second["proxy_target"]).abs().max().item())
        mask_delta = float((first["mask"] - second["mask"]).abs().max().item())
        checks.append(
            {
                "pair_id": row.get("pair_id"),
                "case_id": row["case_id"],
                "target_severity": target_severity,
                "max_source_abs_delta": source_delta,
                "max_proxy_abs_delta": proxy_delta,
                "max_mask_abs_delta": mask_delta,
                "deterministic": source_delta == 0.0 and proxy_delta == 0.0 and mask_delta == 0.0,
            }
        )
    return {
        "sample_count": len(checks),
        "all_deterministic": all(check["deterministic"] for check in checks),
        "checks": checks,
    }


class Exp06ProxyDataset(Dataset[dict[str, Any]]):
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
        target_severity = int(float(row["target_severity"]))
        item = _prepare_proxy_pair(row, target_severity, self.image_size, self.classes)
        source = item["source"]
        proxy = item["proxy_target"]
        mask = item["mask"]
        if self.augment and random.random() < 0.5:
            source = torch.flip(source, dims=[2])
            proxy = torch.flip(proxy, dims=[2])
            mask = torch.flip(mask, dims=[2])
        delta = float(target_severity - item["source_severity"])
        return {
            "source": source,
            "proxy_target": proxy,
            "mask": mask,
            "severity_delta": torch.tensor([delta], dtype=torch.float32),
            "source_severity": torch.tensor(item["source_severity"], dtype=torch.long),
            "target_severity": torch.tensor(target_severity, dtype=torch.long),
            "prompt": _severity_prompt(item["source_severity"], target_severity),
            "case_id": row["case_id"],
            "dataset_key": row["dataset_key"],
        }


class SeverityDeltaProjection(nn.Module):
    def __init__(self, cross_attention_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.net(delta).unsqueeze(1)


class ScalpRegionDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([image * mask, mask], dim=1)).flatten()


def _sampling_weights(rows: list[dict[str, str]]) -> list[float]:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["dataset_key"], str(row["target_severity"]))
        counts[key] = counts.get(key, 0) + 1
    return [1.0 / counts[(row["dataset_key"], str(row["target_severity"]))] for row in rows]


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def _preprocess_for_vae(images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (images.to(dtype=dtype) * 2.0 - 1.0).clamp(-1.0, 1.0)


def _encode_prompt(tokenizer, text_encoder, prompts: list[str], device: torch.device) -> torch.Tensor:
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    return text_encoder(input_ids, attention_mask=attention_mask)[0]


def _load_models(
    model_id: str,
    controlnet_id: str,
    device: torch.device,
    dtype: torch.dtype,
    train_controlnet: bool,
    lora_rank: int,
    lora_alpha: int,
) -> dict[str, Any]:
    from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    vae.to(device).requires_grad_(False)
    text_encoder.to(device).requires_grad_(False)
    unet.to(device)
    controlnet.to(device)

    for parameter in unet.parameters():
        parameter.requires_grad_(False)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    for name, parameter in unet.named_parameters():
        parameter.requires_grad_("lora" in name.lower())

    controlnet.requires_grad_(train_controlnet)
    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "controlnet": controlnet,
        "noise_scheduler": noise_scheduler,
    }


def _predict_x0(
    noisy_latents: torch.Tensor,
    model_pred: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler,
) -> torch.Tensor:
    alphas = noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha_t = alphas[timesteps].view(-1, 1, 1, 1)
    beta_t = 1.0 - alpha_t
    prediction_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        return (noisy_latents - beta_t.sqrt() * model_pred) / alpha_t.sqrt()
    if prediction_type == "v_prediction":
        return alpha_t.sqrt() * noisy_latents - beta_t.sqrt() * model_pred
    return model_pred


def _train(
    models: dict[str, Any],
    loader: DataLoader,
    severity_projection: SeverityDeltaProjection,
    discriminator: ScalpRegionDiscriminator | None,
    device: torch.device,
    dtype: torch.dtype,
    epochs: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    mask_loss_weight: float,
    source_preserve_weight: float,
    guard_weight: float,
) -> list[dict[str, float]]:
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    vae = models["vae"]
    unet = models["unet"]
    controlnet = models["controlnet"]
    noise_scheduler = models["noise_scheduler"]

    trainable_params = (
        [p for p in unet.parameters() if p.requires_grad]
        + [p for p in controlnet.parameters() if p.requires_grad]
        + list(severity_projection.parameters())
    )
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
    disc_optimizer = AdamW(discriminator.parameters(), lr=learning_rate, weight_decay=1e-4) if discriminator else None
    history: list[dict[str, float]] = []
    scaling_factor = vae.config.scaling_factor

    for epoch in range(1, epochs + 1):
        unet.train()
        controlnet.train()
        severity_projection.train()
        if discriminator:
            discriminator.train()
        total_loss = 0.0
        total_guard = 0.0
        total_examples = 0
        optimizer.zero_grad()
        if disc_optimizer:
            disc_optimizer.zero_grad()

        for step, batch in enumerate(loader, start=1):
            source = batch["source"].to(device)
            proxy = batch["proxy_target"].to(device)
            mask = batch["mask"].to(device)
            delta = batch["severity_delta"].to(device, dtype=dtype)
            prompts = list(batch["prompt"])
            with torch.no_grad():
                target_latents = vae.encode(_preprocess_for_vae(proxy, dtype)).latent_dist.sample() * scaling_factor
                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (target_latents.shape[0],),
                    device=device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
                text_embeds = _encode_prompt(tokenizer, text_encoder, prompts, device)
                control_hint = _soft_edge_hint(source, mask).to(device=device, dtype=dtype)

            severity_token = severity_projection(delta)
            encoder_hidden_states = torch.cat([text_embeds, severity_token], dim=1)
            down_residuals, mid_residual = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=control_hint,
                return_dict=False,
            )
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_residuals,
                mid_block_additional_residual=mid_residual,
            ).sample
            denoise_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none").mean(dim=(1, 2, 3))
            latent_mask = F.interpolate(mask, size=target_latents.shape[-2:], mode="area")
            mask_weight = 1.0 + mask_loss_weight * latent_mask.flatten(start_dim=1).mean(dim=1)
            loss = (denoise_loss * mask_weight).mean()

            x0 = _predict_x0(noisy_latents, model_pred, timesteps, noise_scheduler)
            decoded = (vae.decode(x0 / scaling_factor).sample.float() + 1.0) / 2.0
            decoded = decoded.clamp(0.0, 1.0)
            mask_3ch = mask.expand_as(decoded)
            loss = loss + mask_loss_weight * F.l1_loss(decoded * mask_3ch, proxy * mask_3ch)
            loss = loss + source_preserve_weight * F.l1_loss(decoded * (1.0 - mask_3ch), source * (1.0 - mask_3ch))

            guard_loss = torch.tensor(0.0, device=device)
            if discriminator and disc_optimizer:
                with torch.no_grad():
                    fake_for_disc = decoded.detach()
                real_logits = discriminator(proxy, mask)
                fake_logits = discriminator(fake_for_disc, mask)
                disc_loss = (
                    F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                    + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                ) * 0.5
                (disc_loss / gradient_accumulation_steps).backward()
                if step % gradient_accumulation_steps == 0 or step == len(loader):
                    disc_optimizer.step()
                    disc_optimizer.zero_grad()

                _set_requires_grad(discriminator, False)
                guard_logits = discriminator(decoded, mask)
                guard_loss = F.binary_cross_entropy_with_logits(guard_logits, torch.ones_like(guard_logits))
                _set_requires_grad(discriminator, True)
                loss = loss + guard_weight * guard_loss

            (loss / gradient_accumulation_steps).backward()
            if step % gradient_accumulation_steps == 0 or step == len(loader):
                optimizer.step()
                optimizer.zero_grad()

            batch_size = source.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            total_guard += float(guard_loss.detach().item()) * batch_size
            total_examples += batch_size

        summary = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_examples, 1),
            "guard_loss": total_guard / max(total_examples, 1),
        }
        history.append(summary)
        print(json.dumps(summary))
    return history


def _score_image(scorer: nn.Module, labels: list[int], image: torch.Tensor) -> dict[str, float | int]:
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


def _feature_model(device: torch.device) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _generation_pipe(models: dict[str, Any], device: torch.device, dtype: torch.dtype):
    from diffusers import DDIMScheduler, StableDiffusionControlNetPipeline

    pipe = StableDiffusionControlNetPipeline(
        vae=models["vae"],
        text_encoder=models["text_encoder"],
        tokenizer=models["tokenizer"],
        unet=models["unet"],
        controlnet=models["controlnet"],
        scheduler=DDIMScheduler.from_config(models["noise_scheduler"].config),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.unet.eval()
    pipe.controlnet.eval()
    return pipe


def _condition_embeds(
    models: dict[str, Any],
    prompt: str,
    delta: float,
    severity_projection: SeverityDeltaProjection,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        text_embeds = _encode_prompt(models["tokenizer"], models["text_encoder"], [prompt], device)
        negative_embeds = _encode_prompt(models["tokenizer"], models["text_encoder"], [""], device)
        delta_tensor = torch.tensor([[delta]], device=device, dtype=dtype)
        severity_token = severity_projection(delta_tensor)
        zero_token = torch.zeros_like(severity_token)
    return torch.cat([text_embeds, severity_token], dim=1), torch.cat([negative_embeds, zero_token], dim=1)


def _evaluate_generated(
    models: dict[str, Any],
    severity_projection: SeverityDeltaProjection,
    scorer: nn.Module,
    scorer_labels: list[int],
    rows: list[dict[str, str]],
    image_size: int,
    classes: list[int],
    device: torch.device,
    dtype: torch.dtype,
    output_contact_sheet: Path | None,
    split_name: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    mask_blend_strength: float,
    min_monotonic_fraction: float,
    min_expected_span: float,
    max_unmasked_delta_threshold: float,
    mask_area_min: float,
    mask_area_max: float,
) -> dict[str, Any]:
    import lpips

    pipe = _generation_pipe(models, device, dtype)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    features = _feature_model(device)
    contact_rows: list[list[Image.Image]] = []
    cases: list[dict[str, Any]] = []
    generated_feature_values: list[torch.Tensor] = []
    proxy_feature_values: list[torch.Tensor] = []
    to_pil = ToPILImage()

    with torch.no_grad():
        for case_index, row in enumerate(rows):
            source_severity = int(float(row["source_severity"]))
            source_item = _prepare_proxy_pair(row, source_severity, image_size, classes, device=device)
            source = source_item["source"].unsqueeze(0)
            mask = source_item["mask"].unsqueeze(0)
            control_hint = _soft_edge_hint(source, mask).to(device=device, dtype=dtype)
            control_pil = to_pil(control_hint[0].float().cpu())
            source_score = _score_image(scorer, scorer_labels, source)
            generated_expected: list[float] = []
            proxy_expected: list[float] = []
            generated_predicted: list[int] = []
            proxy_predicted: list[int] = []
            generated_l1_values: list[float] = []
            generated_lpips_values: list[float] = []
            proxy_lpips_values: list[float] = []
            guard_values: list[float] = []
            max_unmasked = 0.0
            max_masked = 0.0
            proxy_panels = [
                _add_label(
                    _to_pil(source[0]),
                    [f"{split_name} source", f"sev {source_severity}", row["dataset_key"][:24]],
                )
            ]
            generated_panels = [
                _add_label(
                    _to_pil(source[0]),
                    ["exp06", f"source exp {source_score['expected_severity']:.2f}"],
                )
            ]
            for target_index, target_severity in enumerate(classes):
                item = _prepare_proxy_pair(row, target_severity, image_size, classes, device=device)
                proxy = item["proxy_target"].unsqueeze(0)
                if target_severity == source_severity:
                    generated = source.clone()
                else:
                    delta = float(target_severity - source_severity)
                    prompt = _severity_prompt(source_severity, target_severity)
                    prompt_embeds, negative_prompt_embeds = _condition_embeds(
                        models,
                        prompt,
                        delta,
                        severity_projection,
                        device,
                        dtype,
                    )
                    generator = torch.Generator(device=device).manual_seed(seed + case_index * 100 + target_index)
                    image = pipe(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        image=control_pil,
                        height=image_size,
                        width=image_size,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        generator=generator,
                    ).images[0].convert("RGB")
                    edited = ToTensor()(image).unsqueeze(0).to(device)
                    blend = (mask * mask_blend_strength).clamp(0.0, 1.0)
                    generated = (source * (1.0 - blend) + edited * blend).clamp(0.0, 1.0)

                generated_score = _score_image(scorer, scorer_labels, generated)
                proxy_score = _score_image(scorer, scorer_labels, proxy)
                generated_expected.append(float(generated_score["expected_severity"]))
                proxy_expected.append(float(proxy_score["expected_severity"]))
                generated_predicted.append(int(generated_score["predicted_severity"]))
                proxy_predicted.append(int(proxy_score["predicted_severity"]))
                metrics = _visual_delta_metrics(source, generated, mask)
                max_unmasked = max(max_unmasked, metrics["unmasked_mean_absolute_delta"])
                max_masked = max(max_masked, metrics["masked_mean_absolute_delta"])
                generated_l1_values.append(float((generated - proxy).abs().mean().item()))
                generated_lpips_values.append(float(lpips_model(generated * 2 - 1, proxy * 2 - 1).item()))
                proxy_lpips_values.append(float(lpips_model(proxy * 2 - 1, source * 2 - 1).item()))
                guard_values.append(float(((generated - proxy).abs() * mask).sum().item() / (mask.sum().item() * 3 + 1e-8)))
                generated_feature_values.append(features(generated).float().cpu())
                proxy_feature_values.append(features(proxy).float().cpu())
                proxy_panels.append(
                    _add_label(
                        _to_pil(proxy[0]),
                        [f"proxy t{target_severity}", f"pred {proxy_score['predicted_severity']}", f"exp {proxy_score['expected_severity']:.2f}"],
                    )
                )
                generated_panels.append(
                    _add_label(
                        _to_pil(generated[0]),
                        [f"exp06 t{target_severity}", f"pred {generated_score['predicted_severity']}", f"exp {generated_score['expected_severity']:.2f}"],
                    )
                )

            generated_monotonic = _monotonic_fraction(generated_expected)
            proxy_monotonic = _monotonic_fraction(proxy_expected)
            generated_span = max(generated_expected) - min(generated_expected)
            proxy_span = max(proxy_expected) - min(proxy_expected)
            mask_area = float(mask.mean().item())
            generated_auto_pass = _case_passes(
                generated_monotonic,
                generated_span,
                max_unmasked,
                mask_area,
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
                mask_area,
                min_monotonic_fraction,
                min_expected_span,
                max_unmasked_delta_threshold,
                mask_area_min,
                mask_area_max,
            )
            cases.append(
                {
                    "case_id": row["case_id"],
                    "dataset_key": row["dataset_key"],
                    "source_severity": source_severity,
                    "proxy_expected_severities": proxy_expected,
                    "generated_expected_severities": generated_expected,
                    "proxy_predicted_severities": proxy_predicted,
                    "generated_predicted_severities": generated_predicted,
                    "proxy_monotonic_fraction": proxy_monotonic,
                    "generated_monotonic_fraction": generated_monotonic,
                    "proxy_expected_span": proxy_span,
                    "generated_expected_span": generated_span,
                    "generated_mean_l1_to_proxy": mean(generated_l1_values),
                    "generated_mean_lpips_to_proxy": mean(generated_lpips_values),
                    "proxy_mean_lpips_to_source": mean(proxy_lpips_values),
                    "anatomical_guard_masked_l1_to_proxy": mean(guard_values),
                    "generated_max_masked_mean_absolute_delta": max_masked,
                    "generated_max_unmasked_mean_absolute_delta": max_unmasked,
                    "mask_area_fraction": mask_area,
                    "proxy_auto_pass": proxy_auto_pass,
                    "generated_auto_pass": generated_auto_pass,
                }
            )
            contact_rows.extend([proxy_panels, generated_panels])

    if output_contact_sheet is not None and contact_rows:
        output_contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        _compose_rows(contact_rows).save(output_contact_sheet)

    generated_pass_count = sum(case["generated_auto_pass"] for case in cases)
    proxy_pass_count = sum(case["proxy_auto_pass"] for case in cases)
    generated_features = torch.cat(generated_feature_values, dim=0)
    proxy_features = torch.cat(proxy_feature_values, dim=0)
    by_dataset: dict[str, dict[str, float]] = {}
    for dataset_key in sorted({case["dataset_key"] for case in cases}):
        subset = [case for case in cases if case["dataset_key"] == dataset_key]
        by_dataset[dataset_key] = {
            "case_count": len(subset),
            "mean_generated_expected_span": mean([case["generated_expected_span"] for case in subset]),
            "mean_generated_monotonic_fraction": mean([case["generated_monotonic_fraction"] for case in subset]),
            "generated_auto_pass_rate": sum(case["generated_auto_pass"] for case in subset) / max(len(subset), 1),
        }
    return {
        "split": split_name,
        "case_count": len(cases),
        "generated_auto_pass_count": generated_pass_count,
        "generated_auto_pass_rate": generated_pass_count / max(len(cases), 1),
        "proxy_auto_pass_count": proxy_pass_count,
        "proxy_auto_pass_rate": proxy_pass_count / max(len(cases), 1),
        "mean_generated_monotonic_fraction": mean([case["generated_monotonic_fraction"] for case in cases]),
        "mean_proxy_monotonic_fraction": mean([case["proxy_monotonic_fraction"] for case in cases]),
        "mean_generated_expected_span": mean([case["generated_expected_span"] for case in cases]),
        "mean_proxy_expected_span": mean([case["proxy_expected_span"] for case in cases]),
        "mean_generated_l1_to_proxy": mean([case["generated_mean_l1_to_proxy"] for case in cases]),
        "mean_generated_lpips_to_proxy": mean([case["generated_mean_lpips_to_proxy"] for case in cases]),
        "mean_proxy_lpips_to_source": mean([case["proxy_mean_lpips_to_source"] for case in cases]),
        "domain_fid_resnet18_proxy_reference": _frechet_distance(generated_features, proxy_features),
        "mean_anatomical_guard_masked_l1_to_proxy": mean([case["anatomical_guard_masked_l1_to_proxy"] for case in cases]),
        "mean_generated_max_unmasked_delta": mean([case["generated_max_unmasked_mean_absolute_delta"] for case in cases]),
        "by_dataset": by_dataset,
        "cases": cases,
        "contact_sheet": str(output_contact_sheet) if output_contact_sheet else None,
    }


def _evaluate_proxy_only(
    scorer: nn.Module,
    scorer_labels: list[int],
    rows: list[dict[str, str]],
    image_size: int,
    classes: list[int],
    device: torch.device,
    output_contact_sheet: Path | None,
    split_name: str,
    min_monotonic_fraction: float,
    min_expected_span: float,
    max_unmasked_delta_threshold: float,
    mask_area_min: float,
    mask_area_max: float,
) -> dict[str, Any]:
    contact_rows: list[list[Image.Image]] = []
    cases: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            source_severity = int(float(row["source_severity"]))
            source_item = _prepare_proxy_pair(row, source_severity, image_size, classes, device=device)
            source = source_item["source"].unsqueeze(0)
            mask = source_item["mask"].unsqueeze(0)
            expected_values: list[float] = []
            panels = [
                _add_label(
                    _to_pil(source[0]),
                    [f"{split_name} source", f"sev {source_severity}", row["dataset_key"][:24]],
                )
            ]
            max_unmasked = 0.0
            for target_severity in classes:
                item = _prepare_proxy_pair(row, target_severity, image_size, classes, device=device)
                proxy = item["proxy_target"].unsqueeze(0)
                score = _score_image(scorer, scorer_labels, proxy)
                expected_values.append(float(score["expected_severity"]))
                metrics = _visual_delta_metrics(source, proxy, mask)
                max_unmasked = max(max_unmasked, metrics["unmasked_mean_absolute_delta"])
                panels.append(
                    _add_label(
                        _to_pil(proxy[0]),
                        [f"proxy t{target_severity}", f"pred {score['predicted_severity']}", f"exp {score['expected_severity']:.2f}"],
                    )
                )
            monotonic = _monotonic_fraction(expected_values)
            span = max(expected_values) - min(expected_values)
            mask_area = float(mask.mean().item())
            cases.append(
                {
                    "case_id": row["case_id"],
                    "dataset_key": row["dataset_key"],
                    "source_severity": source_severity,
                    "expected_severities": expected_values,
                    "monotonic_fraction": monotonic,
                    "expected_span": span,
                    "max_unmasked_mean_absolute_delta": max_unmasked,
                    "mask_area_fraction": mask_area,
                    "auto_pass": _case_passes(
                        monotonic,
                        span,
                        max_unmasked,
                        mask_area,
                        min_monotonic_fraction,
                        min_expected_span,
                        max_unmasked_delta_threshold,
                        mask_area_min,
                        mask_area_max,
                    ),
                }
            )
            contact_rows.append(panels)
    if output_contact_sheet and contact_rows:
        output_contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        _compose_rows(contact_rows).save(output_contact_sheet)
    return {
        "split": split_name,
        "case_count": len(cases),
        "proxy_auto_pass_count": sum(case["auto_pass"] for case in cases),
        "proxy_auto_pass_rate": sum(case["auto_pass"] for case in cases) / max(len(cases), 1),
        "mean_proxy_monotonic_fraction": mean([case["monotonic_fraction"] for case in cases]),
        "mean_proxy_expected_span": mean([case["expected_span"] for case in cases]),
        "cases": cases,
        "contact_sheet": str(output_contact_sheet) if output_contact_sheet else None,
    }


def _save_checkpoint(
    output_dir: Path,
    models: dict[str, Any],
    severity_projection: SeverityDeltaProjection,
    discriminator: ScalpRegionDiscriminator | None,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unet_lora_state = {
        key: value.detach().cpu()
        for key, value in models["unet"].state_dict().items()
        if "lora" in key.lower()
    }
    torch.save(unet_lora_state, output_dir / "unet_lora_state_dict.pt")
    torch.save(severity_projection.state_dict(), output_dir / "severity_projection.pt")
    if any(parameter.requires_grad for parameter in models["controlnet"].parameters()):
        torch.save(models["controlnet"].state_dict(), output_dir / "controlnet_state_dict.pt")
    if discriminator:
        torch.save(discriminator.state_dict(), output_dir / "scalp_region_discriminator.pt")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _write_calibration_log(
    checkpoint_dir: Path,
    args: Any,
    device: torch.device,
    rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    classes: list[int],
) -> dict[str, Any]:
    scorer, scorer_labels, scorer_image_size = _load_scorer(args.scorer_checkpoint, device)
    if scorer_image_size != args.image_size:
        raise ValueError(
            f"Scorer image size {scorer_image_size} does not match --image-size {args.image_size}. "
            "Use the scorer's trained image size or rebuild the scorer for this resolution."
        )
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    sample_dir = args.sample_output_dir
    proxy_determinism = _check_proxy_determinism(rows, args.image_size, classes)
    if not proxy_determinism["all_deterministic"]:
        raise RuntimeError(
            "Proxy generation is not deterministic for the sampled rows. "
            "Pre-render proxy targets before training or inspect proxy_parameters."
        )
    val_proxy = _evaluate_proxy_only(
        scorer,
        scorer_labels,
        _unique_case_rows(val_rows),
        args.image_size,
        classes,
        device,
        sample_dir / "val_proxy_contact_sheet.png" if sample_dir else None,
        "val",
        args.min_monotonic_fraction,
        args.min_expected_span,
        args.max_unmasked_delta,
        args.mask_area_min,
        args.mask_area_max,
    )
    test_proxy = _evaluate_proxy_only(
        scorer,
        scorer_labels,
        _unique_case_rows(test_rows),
        args.image_size,
        classes,
        device,
        sample_dir / "test_proxy_contact_sheet.png" if sample_dir else None,
        "test",
        args.min_monotonic_fraction,
        args.min_expected_span,
        args.max_unmasked_delta,
        args.mask_area_min,
        args.mask_area_max,
    )
    calibration = {
        "experiment": "exp06_domain_conditional_diffusion",
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "variant": args.variant,
        "scorer_checkpoint": str(args.scorer_checkpoint),
        "scorer_image_size": scorer_image_size,
        "image_size": args.image_size,
        "proxy_determinism": proxy_determinism,
        "val_proxy_reference": val_proxy,
        "test_proxy_reference": test_proxy,
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "exp06_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    return calibration


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("proxy_pair_manifest", type=Path)
    parser.add_argument("--scorer-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--sample-output-dir", type=Path, default=None)
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--controlnet-id", default="lllyasviel/control_v11p_sd21_softedge")
    parser.add_argument(
        "--variant",
        choices=["severity_delta_cross_attention", "severity_delta_controlnet_guard"],
        default="severity_delta_cross_attention",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--train-controlnet", action="store_true")
    parser.add_argument("--mask-loss-weight", type=float, default=0.5)
    parser.add_argument("--source-preserve-weight", type=float, default=0.3)
    parser.add_argument("--guard-weight", type=float, default=0.2)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--mask-blend-strength", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--min-monotonic-fraction", type=float, default=0.8)
    parser.add_argument("--min-expected-span", type=float, default=0.5)
    parser.add_argument("--max-unmasked-delta", type=float, default=0.01)
    parser.add_argument("--mask-area-min", type=float, default=0.2)
    parser.add_argument("--mask-area-max", type=float, default=0.75)
    args = parser.parse_args()

    _seed_everything(args.seed)
    rows = _load_pair_rows(args.proxy_pair_manifest)
    classes = _condition_classes(rows)
    train_rows = _split_rows(rows, "train")
    val_rows = _split_rows(rows, "val")
    test_rows = _split_rows(rows, "test")
    if args.sample_output_dir:
        args.sample_output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = Exp06ProxyDataset(train_rows, args.image_size, classes, augment=True)
    sampler = WeightedRandomSampler(
        torch.tensor(_sampling_weights(train_rows), dtype=torch.double),
        len(train_rows),
        replacement=True,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    calibration = _write_calibration_log(
        args.checkpoint_dir,
        args,
        device,
        rows,
        val_rows,
        test_rows,
        classes,
    )
    if args.calibration_only:
        summary = {
            "experiment": "exp06_domain_conditional_diffusion",
            "status": "calibration_only_completed",
            "variant": args.variant,
            "proxy_pair_manifest": str(args.proxy_pair_manifest),
            "scorer_checkpoint": str(args.scorer_checkpoint),
            "device": str(device),
            "dtype": str(dtype),
            "classes": classes,
            "image_size": args.image_size,
            "train_pair_count": len(train_rows),
            "val_case_count": len(_unique_case_rows(val_rows)),
            "test_case_count": len(_unique_case_rows(test_rows)),
            "calibration_log": str(args.checkpoint_dir / "exp06_calibration.json"),
            "proxy_determinism": calibration["proxy_determinism"],
            "val_proxy_reference": calibration["val_proxy_reference"],
            "test_proxy_reference": calibration["test_proxy_reference"],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return
    models = _load_models(
        args.model_id,
        args.controlnet_id,
        device,
        dtype,
        args.train_controlnet,
        args.lora_rank,
        args.lora_alpha,
    )
    severity_projection = SeverityDeltaProjection(models["unet"].config.cross_attention_dim).to(device=device, dtype=dtype)
    discriminator = (
        ScalpRegionDiscriminator().to(device)
        if args.variant == "severity_delta_controlnet_guard"
        else None
    )
    trainable_parameters = (
        sum(p.numel() for p in models["unet"].parameters() if p.requires_grad)
        + sum(p.numel() for p in models["controlnet"].parameters() if p.requires_grad)
        + sum(p.numel() for p in severity_projection.parameters() if p.requires_grad)
    )
    history = _train(
        models,
        loader,
        severity_projection,
        discriminator,
        device,
        dtype,
        args.epochs,
        args.learning_rate,
        args.gradient_accumulation_steps,
        args.mask_loss_weight,
        args.source_preserve_weight,
        args.guard_weight,
    )
    _save_checkpoint(
        args.checkpoint_dir,
        models,
        severity_projection,
        discriminator,
        {
            "experiment": "exp06_domain_conditional_diffusion",
            "variant": args.variant,
            "model_id": args.model_id,
            "controlnet_id": args.controlnet_id,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "train_controlnet": args.train_controlnet,
        },
    )
    scorer, scorer_labels, scorer_image_size = _load_scorer(args.scorer_checkpoint, device)
    if scorer_image_size != args.image_size:
        raise ValueError(
            f"Scorer image size {scorer_image_size} does not match --image-size {args.image_size}. "
            "Use the scorer's trained image size or rebuild the scorer for this resolution."
        )
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    sample_dir = args.sample_output_dir
    val_sweep = _evaluate_generated(
        models=models,
        severity_projection=severity_projection,
        scorer=scorer,
        scorer_labels=scorer_labels,
        rows=_unique_case_rows(val_rows),
        image_size=args.image_size,
        classes=classes,
        device=device,
        dtype=dtype,
        output_contact_sheet=sample_dir / "val_exp06_vs_proxy_contact_sheet.png" if sample_dir else None,
        split_name="val",
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        mask_blend_strength=args.mask_blend_strength,
        min_monotonic_fraction=args.min_monotonic_fraction,
        min_expected_span=args.min_expected_span,
        max_unmasked_delta_threshold=args.max_unmasked_delta,
        mask_area_min=args.mask_area_min,
        mask_area_max=args.mask_area_max,
    )
    test_sweep = _evaluate_generated(
        models=models,
        severity_projection=severity_projection,
        scorer=scorer,
        scorer_labels=scorer_labels,
        rows=_unique_case_rows(test_rows),
        image_size=args.image_size,
        classes=classes,
        device=device,
        dtype=dtype,
        output_contact_sheet=sample_dir / "test_exp06_vs_proxy_contact_sheet.png" if sample_dir else None,
        split_name="test",
        seed=args.seed + 1000,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        mask_blend_strength=args.mask_blend_strength,
        min_monotonic_fraction=args.min_monotonic_fraction,
        min_expected_span=args.min_expected_span,
        max_unmasked_delta_threshold=args.max_unmasked_delta,
        mask_area_min=args.mask_area_min,
        mask_area_max=args.mask_area_max,
    )

    summary = {
        "experiment": "exp06_domain_conditional_diffusion",
        "status": "completed",
        "variant": args.variant,
        "model_id": args.model_id,
        "controlnet_id": args.controlnet_id,
        "proxy_pair_manifest": str(args.proxy_pair_manifest),
        "scorer_checkpoint": str(args.scorer_checkpoint),
        "device": str(device),
        "dtype": str(dtype),
        "classes": classes,
        "image_size": args.image_size,
        "train_pair_count": len(train_rows),
        "val_case_count": len(_unique_case_rows(val_rows)),
        "test_case_count": len(_unique_case_rows(test_rows)),
        "trainable_parameters": trainable_parameters,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "history": history,
        },
        "generation": {
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "controlnet_conditioning_scale": args.controlnet_conditioning_scale,
            "mask_blend_strength": args.mask_blend_strength,
            "seed": args.seed,
        },
        "calibration_log": str(args.checkpoint_dir / "exp06_calibration.json"),
        "proxy_determinism": calibration["proxy_determinism"],
        "qa_criteria": {
            "min_monotonic_fraction": args.min_monotonic_fraction,
            "min_expected_span": args.min_expected_span,
            "max_unmasked_delta": args.max_unmasked_delta,
            "mask_area_min": args.mask_area_min,
            "mask_area_max": args.mask_area_max,
        },
        "val_proxy_reference": calibration["val_proxy_reference"],
        "test_proxy_reference": calibration["test_proxy_reference"],
        "val_sweep": val_sweep,
        "test_sweep": test_sweep,
        "limitations": [
            "The anatomical guard discriminator is weakly supervised by QA-gated proxy targets and should be supplemented by contact-sheet review or classifier labels.",
            "The FID-like metric uses ImageNet ResNet18 features against proxy references rather than a fully trained scalp-domain FID model.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
