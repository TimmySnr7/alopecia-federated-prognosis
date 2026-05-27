"""Run matched Exp09 perturbation evaluation for the Exp06 diffusion editor.

The original Exp09 report used the Exp06 severity-delta diffusion result as an
archived reference. This script loads the saved Exp06 LoRA and severity
projection artefacts and re-runs diffusion inference under the same perturbation
families used for the proxy and sparse-controller pipelines.
"""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torchvision.transforms import InterpolationMode, Resize, ToPILImage, ToTensor

from experiments.scripts.run_exp01_plausible_proxy_sweep import _add_label, _monotonic_fraction, _to_pil
from experiments.scripts.run_exp01_proxy_batch_eval import _load_scorer, _visual_delta_metrics
from experiments.scripts.run_exp09_fragility_evaluation import (
    TARGET_LABELS,
    WEIGHT_SETTINGS,
    _case_metrics,
    _fragility,
    _perturbation_specs,
    _score_components,
    _stable_seed,
)
from experiments.scripts.train_exp03_proxy_residual_editor import (
    _case_passes,
    _compose_rows,
    _condition_classes,
    _load_pair_rows,
    _unique_case_rows,
)
from experiments.scripts.train_exp06_domain_conditional_diffusion import (
    SeverityDeltaProjection,
    _condition_embeds,
    _generation_pipe,
    _load_models,
    _prepare_proxy_pair,
    _score_image,
    _severity_prompt,
    _soft_edge_hint,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys and key != "cases":
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _rows_by_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    seen: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["split"] == split:
            seen.setdefault(row["case_id"], row)
    return list(seen.values())


def _nearest_target(value: float, targets: list[int]) -> int:
    return min(targets, key=lambda target: abs(float(target) - value))


def _perturb_image(image: Image.Image, spec: dict[str, Any], rng: np.random.Generator) -> Image.Image:
    if spec["family"] == "brightness":
        return ImageEnhance.Brightness(image).enhance(1.0 + float(spec["value"]))
    if spec["family"] == "contrast":
        return ImageEnhance.Contrast(image).enhance(1.0 + float(spec["value"]))
    if spec["family"] == "image_noise":
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0.0, float(spec["value"]), size=arr.shape), 0.0, 1.0)
        return Image.fromarray(np.uint8(np.round(arr * 255.0)), mode="RGB")
    return image


def _perturb_row(row: dict[str, str], spec: dict[str, Any], image_size: int) -> dict[str, str]:
    item = dict(row)
    family = spec["family"]
    if family not in {"mask_erode", "mask_dilate", "mask_shift"}:
        return item
    ellipse = [float(value) for value in json.loads(item["ellipse_mask"])]
    px = float(spec.get("px", 0.0)) / float(image_size)
    if family == "mask_erode":
        ellipse[2] = max(0.05, ellipse[2] - px)
        ellipse[3] = max(0.05, ellipse[3] - px)
    elif family == "mask_dilate":
        ellipse[2] = min(0.60, ellipse[2] + px)
        ellipse[3] = min(0.70, ellipse[3] + px)
    elif family == "mask_shift":
        ellipse[0] = min(0.95, max(0.05, ellipse[0] + float(spec["dx"]) * px))
        ellipse[1] = min(0.95, max(0.05, ellipse[1] + float(spec["dy"]) * px))
    item["ellipse_mask"] = json.dumps(ellipse)
    params = json.loads(item["proxy_parameters"])
    params["ellipse_mask"] = ellipse
    item["proxy_parameters"] = json.dumps(params)
    return item


def _load_trained_exp06(
    checkpoint_dir: Path,
    model_id: str,
    controlnet_id: str,
    device: torch.device,
    dtype: torch.dtype,
    lora_rank: int,
    lora_alpha: int,
) -> tuple[dict[str, Any], SeverityDeltaProjection]:
    models = _load_models(
        model_id=model_id,
        controlnet_id=controlnet_id,
        device=device,
        dtype=dtype,
        train_controlnet=False,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    unet_state = torch.load(checkpoint_dir / "unet_lora_state_dict.pt", map_location="cpu")
    missing, unexpected = models["unet"].load_state_dict(unet_state, strict=False)
    severe_unexpected = [key for key in unexpected if "lora" in key.lower()]
    if severe_unexpected:
        raise RuntimeError(f"Unexpected LoRA keys while loading Exp06 checkpoint: {severe_unexpected[:8]}")
    severity_projection = SeverityDeltaProjection(models["unet"].config.cross_attention_dim).to(device=device, dtype=dtype)
    severity_projection.load_state_dict(torch.load(checkpoint_dir / "severity_projection.pt", map_location=device))
    severity_projection.eval()
    for module in [models["vae"], models["text_encoder"], models["unet"], models["controlnet"], severity_projection]:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    if missing:
        print(json.dumps({"load_note": "Exp06 LoRA loaded with non-LoRA base-model missing keys", "missing_count": len(missing)}))
    return models, severity_projection


def _evaluate_diffusion(
    split: str,
    spec: dict[str, Any],
    cases: list[dict[str, str]],
    target_labels: list[int],
    models: dict[str, Any],
    severity_projection: SeverityDeltaProjection,
    scorer: torch.nn.Module,
    labels: list[int],
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    mask_blend_strength: float,
    make_contact: bool,
) -> dict[str, Any]:
    pipe = _generation_pipe(models, device, dtype)
    rng = np.random.default_rng(seed)
    to_pil = ToPILImage()
    result_cases: list[dict[str, Any]] = []
    contact_rows: list[list[Image.Image]] = []
    resize = Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
    with torch.no_grad():
        for case_index, base_row in enumerate(sorted(cases, key=lambda row: row["case_id"])):
            row = _perturb_row(base_row, spec, image_size)
            source_severity = int(float(row["source_severity"]))
            source_item = _prepare_proxy_pair(row, source_severity, image_size, target_labels, device=device)
            source_image = _perturb_image(source_item["source_pil"], spec, rng)
            source = ToTensor()(source_image).unsqueeze(0).to(device)
            mask = source_item["mask"].unsqueeze(0)
            control_hint = _soft_edge_hint(source, mask).to(device=device, dtype=dtype)
            control_pil = to_pil(control_hint[0].float().cpu())
            source_score = _score_image(scorer, labels, source)
            expected: list[float] = []
            panels = [
                _add_label(
                    _to_pil(source[0]),
                    [f"P2 {spec['id']}", f"source {source_severity}", f"exp {source_score['expected_severity']:.2f}"],
                )
            ]
            max_unmasked = 0.0
            guard_values: list[float] = []
            for target_index, target in enumerate(target_labels):
                target_float = float(target)
                if spec["family"] == "target_noise":
                    target_float = float(np.clip(target_float + rng.normal(0.0, float(spec["value"])), 1.0, 7.0))
                prompt_target = _nearest_target(target_float, target_labels)
                proxy_item = _prepare_proxy_pair(row, prompt_target, image_size, target_labels, device=device)
                proxy = proxy_item["proxy_target"].unsqueeze(0)
                if abs(target_float - float(source_severity)) < 1e-6:
                    generated = source.clone()
                else:
                    prompt = _severity_prompt(source_severity, prompt_target)
                    prompt_embeds, negative_prompt_embeds = _condition_embeds(
                        models,
                        prompt,
                        float(target_float - source_severity),
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
                score = _score_image(scorer, labels, generated)
                visual = _visual_delta_metrics(source, generated, mask)
                guard_l1 = float(((generated - proxy).abs() * mask).sum().item() / (mask.sum().item() * 3 + 1e-8))
                guard_values.append(guard_l1)
                max_unmasked = max(max_unmasked, float(visual["unmasked_mean_absolute_delta"]))
                expected.append(float(score["expected_severity"]))
                panels.append(
                    _add_label(
                        _to_pil(generated[0]),
                        [f"target {target_float:.2f}", f"exp {score['expected_severity']:.2f}", f"drift {visual['unmasked_mean_absolute_delta']:.4f}"],
                    )
                )
            sev = _case_metrics(expected)
            mask_area = float(mask.mean().item())
            scalp_valid = max_unmasked <= 0.01 and 0.2 <= mask_area <= 0.75
            auto_pass = _case_passes(
                float(sev["monotonic_fraction"]),
                float(sev["expected_span"]),
                max_unmasked,
                mask_area,
                0.8,
                0.5,
                0.01,
                0.2,
                0.75,
            )
            mean_guard_l1 = float(mean(guard_values)) if guard_values else 0.0
            critical_error = bool((not scalp_valid) or mean_guard_l1 > 0.12)
            result_cases.append(
                {
                    "case_id": row["case_id"],
                    "expected_severities": expected,
                    **sev,
                    "max_unmasked_delta": max_unmasked,
                    "mask_area_fraction": mask_area,
                    "scalp_localization_valid": scalp_valid,
                    "auto_pass": auto_pass,
                    "mean_guard_l1_to_proxy": mean_guard_l1,
                    "critical_error": critical_error,
                }
            )
            if make_contact:
                contact_rows.append(panels)
    components = _score_components(result_cases)
    result = {
        "pipeline_id": "P2_diffusion_exp06_rerun",
        "split": split,
        "perturbation": spec["id"],
        "perturbation_family": spec["family"],
        "perturbation_level": spec["level"],
        "case_count": len(result_cases),
        "auto_pass_rate": float(np.mean([case["auto_pass"] for case in result_cases])),
        "mean_monotonic_fraction": float(np.mean([case["monotonic_fraction"] for case in result_cases])),
        "mean_violation_count": float(np.mean([case["violation_count"] for case in result_cases])),
        "mean_spearman_rho": float(np.mean([case["spearman_rho"] for case in result_cases])),
        "mean_expected_span": float(np.mean([case["expected_span"] for case in result_cases])),
        "mean_max_unmasked_delta": float(np.mean([case["max_unmasked_delta"] for case in result_cases])),
        "mean_mask_area_fraction": float(np.mean([case["mask_area_fraction"] for case in result_cases])),
        "mean_guard_l1_to_proxy": float(np.mean([case["mean_guard_l1_to_proxy"] for case in result_cases])),
        **components,
        "fragility_balanced": _fragility(components, WEIGHT_SETTINGS["balanced"]),
        "cases": result_cases,
    }
    if make_contact and contact_rows:
        path = output_dir / f"{split}_P2_diffusion_exp06_rerun_{spec['id']}_contact_sheet.png"
        _compose_rows(contact_rows).save(path)
        result["contact_sheet"] = str(path)
    return result


def _summaries(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    family_rows = []
    component_rows = []
    weight_rows = []
    families = sorted({row["perturbation_family"] for row in results})
    for family in families:
        subset = [row for row in results if row["perturbation_family"] == family]
        family_rows.append(
            {
                "pipeline_id": "P2_diffusion_exp06_rerun",
                "perturbation_family": family,
                "mean_fragility_balanced": float(np.mean([row["fragility_balanced"] for row in subset])),
                "mean_auto_pass_rate": float(np.mean([row["auto_pass_rate"] for row in subset])),
                "mean_monotonic_fraction": float(np.mean([row["mean_monotonic_fraction"] for row in subset])),
                "mean_critical_error_rate": float(np.mean([row["e_crit"] for row in subset])),
            }
        )
        component_rows.append(
            {
                "pipeline_id": "P2_diffusion_exp06_rerun",
                "perturbation_family": family,
                "s_order": float(np.mean([row["s_order"] for row in subset])),
                "s_qa": float(np.mean([row["s_qa"] for row in subset])),
                "e_crit": float(np.mean([row["e_crit"] for row in subset])),
                "ordering_only": float(np.mean([1.0 - row["s_order"] for row in subset])),
                "qa_only": float(np.mean([1.0 - row["s_qa"] for row in subset])),
                "critical_only": float(np.mean([row["e_crit"] for row in subset])),
            }
        )
    for name, weights in WEIGHT_SETTINGS.items():
        values = [_fragility({"s_order": row["s_order"], "s_qa": row["s_qa"], "e_crit": row["e_crit"]}, weights) for row in results]
        weight_rows.append(
            {
                "pipeline_id": "P2_diffusion_exp06_rerun",
                "weight_setting": name,
                "alpha": weights[0],
                "beta": weights[1],
                "gamma": weights[2],
                "mean_fragility": float(np.mean(values)),
            }
        )
    return family_rows, component_rows, weight_rows


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--pair-manifest", type=Path, default=Path("experiments/results/exp02/source_aware_proxy_batch_v1/proxy_pair_manifest.csv"))
    parser.add_argument("--scorer-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/exp09/diffusion_perturbation_rerun_v1"))
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet-id", default="lllyasviel/control_v11p_sd15_softedge")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--mask-blend-strength", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=9906)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_pair_rows(args.pair_manifest)
    classes = _condition_classes(rows)
    target_labels = [label for label in TARGET_LABELS if label in classes]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    scorer, labels, scorer_image_size = _load_scorer(args.scorer_checkpoint, device)
    if scorer_image_size != args.image_size:
        raise ValueError(f"Scorer image size {scorer_image_size} does not match --image-size {args.image_size}.")
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    models, severity_projection = _load_trained_exp06(
        args.checkpoint_dir,
        args.model_id,
        args.controlnet_id,
        device,
        dtype,
        args.lora_rank,
        args.lora_alpha,
    )
    specs = _perturbation_specs()
    results: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        split_specs = [spec for spec in specs if spec["level"] in {"nominal", "low"}] if split == "val" else specs
        cases = _rows_by_split(rows, split)
        for spec in split_specs:
            make_contact = split == "test" and spec["id"] in {"nominal", "parameter_noise_high", "brightness_up_high", "mask_shift_pos_high"}
            result = _evaluate_diffusion(
                split=split,
                spec=spec,
                cases=cases,
                target_labels=target_labels,
                models=models,
                severity_projection=severity_projection,
                scorer=scorer,
                labels=labels,
                image_size=args.image_size,
                device=device,
                dtype=dtype,
                output_dir=args.output_dir,
                seed=args.seed + _stable_seed(args.seed, split, spec["id"]),
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                mask_blend_strength=args.mask_blend_strength,
                make_contact=make_contact,
            )
            print(json.dumps({key: result[key] for key in result if key != "cases"}, indent=2))
            results.append(result)
    family_rows, component_rows, weight_rows = _summaries(results)
    _write_csv(args.output_dir / "diffusion_fragility_by_perturbation.csv", results)
    _write_csv(args.output_dir / "diffusion_mean_fragility_by_family.csv", family_rows)
    _write_csv(args.output_dir / "diffusion_component_breakdown.csv", component_rows)
    _write_csv(args.output_dir / "diffusion_weight_sensitivity.csv", weight_rows)
    summary = {
        "experiment": "exp09_diffusion_perturbation_rerun_v1",
        "status": "completed_matched_diffusion_perturbation_rerun",
        "checkpoint_dir": str(args.checkpoint_dir),
        "model_id": args.model_id,
        "controlnet_id": args.controlnet_id,
        "target_labels": target_labels,
        "weights": WEIGHT_SETTINGS,
        "critical_error_note": "Critical errors are computational critical-risk heuristics based on scalp localisation and masked L1 to proxy; contact sheets should be reviewed for semantic labels.",
        "results": results,
        "mean_fragility_by_pipeline_family": family_rows,
        "component_breakdown": component_rows,
        "weight_sensitivity": weight_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
