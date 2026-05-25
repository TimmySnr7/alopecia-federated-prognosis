"""Run Experiment 04's pretrained latent image-editing baseline."""

from __future__ import annotations

from argparse import ArgumentParser
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
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


def _load_pipeline(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
):
    try:
        from diffusers import StableDiffusionInstructPix2PixPipeline
        from diffusers.schedulers import EulerAncestralDiscreteScheduler
    except ImportError as exc:
        raise RuntimeError(
            "Experiment 04 requires diffusers with StableDiffusionInstructPix2PixPipeline."
        ) from exc

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def _severity_prompt(source_severity: int, target_severity: int) -> str:
    if target_severity == source_severity:
        return "Preserve the scalp photograph without changing hair density."
    direction = "increase hair density and scalp coverage" if target_severity < source_severity else (
        "reduce hair density and make scalp thinning more visible"
    )
    return (
        "Edit only the scalp hair region in this clinical top-view scalp photograph. "
        f"{direction} so the alopecia severity looks like ordinal stage {target_severity}. "
        "Preserve camera angle, identity, lighting, background, skin tone, and image realism."
    )


def _prepare_source_and_proxy(
    row: dict[str, str],
    target_severity: int,
    image_size: int,
    classes: list[int],
    device: torch.device,
) -> dict[str, Any]:
    source_severity = int(float(row["source_severity"]))
    proxy_parameters = _load_proxy_parameters(row)
    ellipse = _load_ellipse(row)
    image = Image.open(row["image_path"]).convert("RGB")
    resized = Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)(image)
    blurred = resized.filter(
        ImageFilter.GaussianBlur(radius=float(proxy_parameters["texture_blur_radius"]))
    )
    source = ToTensor()(resized).unsqueeze(0).to(device)
    blurred_tensor = ToTensor()(blurred).unsqueeze(0).to(device)
    mask = _build_ellipse_mask(
        ellipse,
        image_size=image_size,
        device=device,
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
        "source": source,
        "mask": mask,
        "proxy_target": proxy_target,
        "source_severity": source_severity,
    }


def _score_image(
    scorer: torch.nn.Module,
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


def _edit_with_pipeline(
    pipe,
    source_pil: Image.Image,
    source: torch.Tensor,
    mask: torch.Tensor,
    prompt: str,
    target_severity: int,
    source_severity: int,
    device: torch.device,
    generator: torch.Generator,
    num_inference_steps: int,
    guidance_scale: float,
    image_guidance_scale: float,
    mask_blend_strength: float,
) -> torch.Tensor:
    if target_severity == source_severity:
        return source.clone()
    edited_pil = pipe(
        prompt=prompt,
        image=source_pil,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        generator=generator,
    ).images[0].convert("RGB")
    edited = ToTensor()(edited_pil).unsqueeze(0).to(device)
    if edited.shape[-2:] != source.shape[-2:]:
        edited = F.interpolate(edited, size=source.shape[-2:], mode="bilinear", align_corners=False)
    edit_mask = (mask * mask_blend_strength).clamp(0.0, 1.0)
    return torch.clamp(source * (1.0 - edit_mask) + edited * edit_mask, 0.0, 1.0)


def _evaluate_cases(
    pipe,
    rows: list[dict[str, str]],
    image_size: int,
    classes: list[int],
    scorer: torch.nn.Module,
    scorer_labels: list[int],
    device: torch.device,
    output_contact_sheet: Path | None,
    split_name: str,
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
    case_summaries: list[dict[str, Any]] = []
    contact_rows: list[list[Image.Image]] = []
    scorer.eval()

    with torch.no_grad():
        for case_index, row in enumerate(rows):
            source_severity = int(float(row["source_severity"]))
            source_item = _prepare_source_and_proxy(
                row=row,
                target_severity=source_severity,
                image_size=image_size,
                classes=classes,
                device=device,
            )
            source = source_item["source"]
            mask = source_item["mask"]
            source_pil = source_item["source_pil"]
            source_score = _score_image(scorer, scorer_labels, source)

            proxy_expected: list[float] = []
            proxy_predicted: list[int] = []
            latent_expected: list[float] = []
            latent_predicted: list[int] = []
            latent_max_unmasked_delta = 0.0
            latent_max_masked_delta = 0.0
            latent_proxy_l1_values: list[float] = []

            proxy_panels: list[Image.Image] = [
                _add_label(
                    _to_pil(source[0]),
                    [f"{split_name} source", f"sev {source_severity}", row["dataset_key"][:24]],
                )
            ]
            latent_panels: list[Image.Image] = [
                _add_label(
                    _to_pil(source[0]),
                    ["latent", f"source exp {source_score['expected_severity']:.2f}"],
                )
            ]

            for target_index, target_severity in enumerate(classes):
                item = _prepare_source_and_proxy(
                    row=row,
                    target_severity=target_severity,
                    image_size=image_size,
                    classes=classes,
                    device=device,
                )
                proxy_target = item["proxy_target"]
                prompt = _severity_prompt(source_severity, target_severity)
                generator = torch.Generator(device=device).manual_seed(
                    seed + case_index * 100 + target_index
                )
                latent = _edit_with_pipeline(
                    pipe=pipe,
                    source_pil=source_pil,
                    source=source,
                    mask=mask,
                    prompt=prompt,
                    target_severity=target_severity,
                    source_severity=source_severity,
                    device=device,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    image_guidance_scale=image_guidance_scale,
                    mask_blend_strength=mask_blend_strength,
                )

                proxy_score = _score_image(scorer, scorer_labels, proxy_target)
                latent_score = _score_image(scorer, scorer_labels, latent)
                proxy_expected.append(float(proxy_score["expected_severity"]))
                proxy_predicted.append(int(proxy_score["predicted_severity"]))
                latent_expected.append(float(latent_score["expected_severity"]))
                latent_predicted.append(int(latent_score["predicted_severity"]))

                latent_metrics = _visual_delta_metrics(source, latent, mask)
                latent_max_unmasked_delta = max(
                    latent_max_unmasked_delta,
                    latent_metrics["unmasked_mean_absolute_delta"],
                )
                latent_max_masked_delta = max(
                    latent_max_masked_delta,
                    latent_metrics["masked_mean_absolute_delta"],
                )
                latent_proxy_l1_values.append(float((latent - proxy_target).abs().mean().item()))

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
                latent_panels.append(
                    _add_label(
                        _to_pil(latent[0]),
                        [
                            f"latent t{target_severity}",
                            f"pred {latent_score['predicted_severity']}",
                            f"exp {latent_score['expected_severity']:.2f}",
                        ],
                    )
                )

            latent_monotonic = _monotonic_fraction(latent_expected)
            proxy_monotonic = _monotonic_fraction(proxy_expected)
            latent_span = max(latent_expected) - min(latent_expected)
            proxy_span = max(proxy_expected) - min(proxy_expected)
            mask_area_fraction = float(mask.mean().item())
            latent_auto_pass = _case_passes(
                monotonic_fraction=latent_monotonic,
                expected_span=latent_span,
                max_unmasked_delta=latent_max_unmasked_delta,
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
                    "latent_expected_severities": latent_expected,
                    "proxy_predicted_severities": proxy_predicted,
                    "latent_predicted_severities": latent_predicted,
                    "proxy_monotonic_fraction": proxy_monotonic,
                    "latent_monotonic_fraction": latent_monotonic,
                    "proxy_expected_span": proxy_span,
                    "latent_expected_span": latent_span,
                    "latent_mean_l1_to_proxy": mean(latent_proxy_l1_values),
                    "latent_max_masked_mean_absolute_delta": latent_max_masked_delta,
                    "latent_max_unmasked_mean_absolute_delta": latent_max_unmasked_delta,
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
    return {
        "split": split_name,
        "case_count": len(case_summaries),
        "latent_auto_pass_count": latent_pass_count,
        "latent_auto_pass_rate": latent_pass_count / max(len(case_summaries), 1),
        "proxy_auto_pass_count": proxy_pass_count,
        "proxy_auto_pass_rate": proxy_pass_count / max(len(case_summaries), 1),
        "mean_latent_monotonic_fraction": mean(
            [case["latent_monotonic_fraction"] for case in case_summaries]
        )
        if case_summaries
        else 0.0,
        "mean_proxy_monotonic_fraction": mean(
            [case["proxy_monotonic_fraction"] for case in case_summaries]
        )
        if case_summaries
        else 0.0,
        "mean_latent_expected_span": mean(
            [case["latent_expected_span"] for case in case_summaries]
        )
        if case_summaries
        else 0.0,
        "mean_proxy_expected_span": mean([case["proxy_expected_span"] for case in case_summaries])
        if case_summaries
        else 0.0,
        "mean_latent_l1_to_proxy": mean(
            [case["latent_mean_l1_to_proxy"] for case in case_summaries]
        )
        if case_summaries
        else 0.0,
        "mean_latent_max_unmasked_delta": mean(
            [case["latent_max_unmasked_mean_absolute_delta"] for case in case_summaries]
        )
        if case_summaries
        else 0.0,
        "cases": case_summaries,
        "contact_sheet": str(output_contact_sheet) if output_contact_sheet else None,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("proxy_pair_manifest", type=Path)
    parser.add_argument("--scorer-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sample-output-dir", type=Path, default=None)
    parser.add_argument("--model-id", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--mask-blend-strength", type=float, default=1.0)
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
    val_rows = _split_rows(rows, "val")
    test_rows = _split_rows(rows, "test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = _load_pipeline(args.model_id, device=device, dtype=dtype)
    scorer, scorer_labels, scorer_image_size = _load_scorer(args.scorer_checkpoint, device)
    if scorer_image_size != args.image_size:
        raise ValueError(
            f"Scorer image size {scorer_image_size} does not match requested {args.image_size}."
        )
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)

    sample_dir = args.sample_output_dir
    val_sweep = _evaluate_cases(
        pipe=pipe,
        rows=_unique_case_rows(val_rows),
        image_size=args.image_size,
        classes=classes,
        scorer=scorer,
        scorer_labels=scorer_labels,
        device=device,
        output_contact_sheet=sample_dir / "val_latent_vs_proxy_contact_sheet.png"
        if sample_dir
        else None,
        split_name="val",
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        mask_blend_strength=args.mask_blend_strength,
        min_monotonic_fraction=args.min_monotonic_fraction,
        min_expected_span=args.min_expected_span,
        max_unmasked_delta_threshold=args.max_unmasked_delta,
        mask_area_min=args.mask_area_min,
        mask_area_max=args.mask_area_max,
    )
    test_sweep = _evaluate_cases(
        pipe=pipe,
        rows=_unique_case_rows(test_rows),
        image_size=args.image_size,
        classes=classes,
        scorer=scorer,
        scorer_labels=scorer_labels,
        device=device,
        output_contact_sheet=sample_dir / "test_latent_vs_proxy_contact_sheet.png"
        if sample_dir
        else None,
        split_name="test",
        seed=args.seed + 10_000,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        mask_blend_strength=args.mask_blend_strength,
        min_monotonic_fraction=args.min_monotonic_fraction,
        min_expected_span=args.min_expected_span,
        max_unmasked_delta_threshold=args.max_unmasked_delta,
        mask_area_min=args.mask_area_min,
        mask_area_max=args.mask_area_max,
    )

    summary = {
        "experiment": "exp04_pretrained_latent_editor",
        "model_id": args.model_id,
        "proxy_pair_manifest": str(args.proxy_pair_manifest),
        "scorer_checkpoint": str(args.scorer_checkpoint),
        "device": str(device),
        "dtype": str(dtype),
        "image_size": args.image_size,
        "classes": classes,
        "val_case_count": len(_unique_case_rows(val_rows)),
        "test_case_count": len(_unique_case_rows(test_rows)),
        "generation": {
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "image_guidance_scale": args.image_guidance_scale,
            "mask_blend_strength": args.mask_blend_strength,
            "seed": args.seed,
            "identity_targets_are_source_passthrough": True,
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
        "sample_output_dir": str(sample_dir) if sample_dir else None,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
