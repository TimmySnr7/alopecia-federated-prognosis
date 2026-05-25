"""Experiment 09: fragility-aware evaluation of alopecia severity synthesis.

This final evaluation stresses the validated proxy renderer and sparse
controllers under small realistic perturbations. The direct diffusion baseline
is included as an archived Exp06 comparator because no lightweight deterministic
diffusion inference artefact is available for per-perturbation re-rendering.
"""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torchvision.transforms import InterpolationMode, Resize, ToTensor

from experiments.scripts.run_exp01_plausible_proxy_sweep import (
    _add_label,
    _build_ellipse_mask,
    _expected_severity,
    _monotonic_fraction,
    _to_pil,
)
from experiments.scripts.run_exp01_proxy_batch_eval import _load_scorer, _visual_delta_metrics
from experiments.scripts.run_exp07_v2_image_aware_proxy_controller import (
    IMAGE_STAT_NAMES,
    PARAMETER_NAMES,
    _clip,
    _compose_grid,
    _fit_model,
    _load_json_cell,
    _load_rows,
    _predict,
    _render,
    _score_image,
    _target_matrix,
)
from experiments.scripts.run_exp08_qa_refined_proxy_controller import SPARSE2_STATS, SPARSE4_STATS, _masked_stats
from experiments.scripts.train_exp07_proxy_parameter_controller import _fit_standardizer


TARGET_LABELS = [1, 2, 4, 6, 7]
WEIGHT_SETTINGS = {
    "balanced": (0.4, 0.4, 0.2),
    "safety_heavy": (0.25, 0.25, 0.5),
    "qa_heavy": (0.3, 0.6, 0.1),
    "ordering_heavy": (0.6, 0.3, 0.1),
}


def _stable_seed(seed: int, *parts: Any) -> int:
    digest = hashlib.sha256("::".join([str(seed), *[str(part) for part in parts]]).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 100000


def _rows_by_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    seen: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["split"] == split:
            seen.setdefault(row["case_id"], row)
    return list(seen.values())


def _eval_rows(case_rows: list[dict[str, str]], targets: list[int]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in case_rows:
        for target in targets:
            item = dict(row)
            item["target_severity"] = str(target)
            item["pair_id"] = f"{row['case_id']}__exp09_target_{target}"
            out.append(item)
    return out


def _load_image(row: dict[str, str], image_size: int) -> Image.Image:
    image = Image.open(row["image_path"]).convert("RGB")
    return Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)(image)


def _image_stats_from_image(row: dict[str, str], image: Image.Image, image_size: int) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=3.0)).convert("L"), dtype=np.float32) / 255.0
    texture = np.abs(gray - blurred)
    ellipse = [float(value) for value in _load_json_cell(row, "ellipse_mask")]
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    cx, cy, rx, ry = ellipse
    hard_mask = (((xx / image_size - cx) / max(rx, 1e-6)) ** 2 + ((yy / image_size - cy) / max(ry, 1e-6)) ** 2) <= 1.0
    inv_mask = ~hard_mask
    masked_gray = gray[hard_mask]
    masked_texture = texture[hard_mask]
    unmasked_gray = gray[inv_mask]
    dark_fraction = float((masked_gray < 0.38).mean()) if masked_gray.size else 0.0
    texture_mean = float(masked_texture.mean()) if masked_texture.size else 0.0
    hair_presence = float(np.clip(0.65 * dark_fraction + 6.0 * texture_mean, 0.0, 1.0))
    return np.asarray(
        [
            float(masked_gray.mean()) if masked_gray.size else 0.0,
            float(masked_gray.std()) if masked_gray.size else 0.0,
            dark_fraction,
            float((masked_gray > 0.72).mean()) if masked_gray.size else 0.0,
            texture_mean,
            float(np.quantile(masked_texture, 0.90)) if masked_texture.size else 0.0,
            float(unmasked_gray.mean()) if unmasked_gray.size else 0.0,
            float(hard_mask.mean()),
            hair_presence,
        ],
        dtype=np.float64,
    )


def _stats_cache_for_rows(rows: list[dict[str, str]], image_size: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for row in rows:
        out.setdefault(row["case_id"], _image_stats_from_image(row, _load_image(row, image_size), image_size))
    return out


def _perturb_image(image: Image.Image, spec: dict[str, Any], rng: np.random.Generator) -> Image.Image:
    family = spec["family"]
    if family == "brightness":
        return ImageEnhance.Brightness(image).enhance(1.0 + float(spec["value"]))
    if family == "contrast":
        return ImageEnhance.Contrast(image).enhance(1.0 + float(spec["value"]))
    if family == "image_noise":
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0.0, float(spec["value"]), size=arr.shape), 0.0, 1.0)
        return Image.fromarray(np.uint8(np.round(arr * 255.0)), mode="RGB")
    return image


def _perturb_eval_row(row: dict[str, str], spec: dict[str, Any], rng: np.random.Generator) -> dict[str, str]:
    item = dict(row)
    if spec["family"] == "target_noise":
        noisy = float(item["target_severity"]) + float(rng.normal(0.0, spec["value"]))
        item["target_severity"] = str(float(np.clip(noisy, 1.0, 7.0)))
    return item


def _perturb_vectors(
    vectors: np.ndarray,
    spec: dict[str, Any],
    image_size: int,
    train_std: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    out = vectors.copy()
    family = spec["family"]
    px = float(spec.get("px", 0.0)) / float(image_size)
    if family == "mask_erode":
        out[:, 2] -= px
        out[:, 3] -= px
    elif family == "mask_dilate":
        out[:, 2] += px
        out[:, 3] += px
    elif family == "mask_shift":
        out[:, 0] += float(spec["dx"]) * px
        out[:, 1] += float(spec["dy"]) * px
    elif family == "parameter_noise":
        editable = np.asarray([6, 7, 8, 9, 10])
        out[:, editable] += rng.normal(0.0, float(spec["value"]), size=(out.shape[0], len(editable))) * train_std[editable]
    return np.vstack([_clip(row) for row in out])


def _case_metrics(expected: list[float]) -> dict[str, float | int]:
    diffs = [b - a for a, b in zip(expected, expected[1:])]
    violations = sum(1 for value in diffs if value < 0)
    values = np.asarray(expected, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    target = np.arange(len(values), dtype=np.float64)
    rho = 0.0 if np.std(ranks) < 1e-8 else float(np.corrcoef(target, ranks)[0, 1])
    return {
        "monotonic_fraction": float(_monotonic_fraction(expected)),
        "violation_count": int(violations),
        "spearman_rho": rho,
        "expected_span": float(max(expected) - min(expected)) if expected else 0.0,
    }


def _score_components(cases: list[dict[str, Any]]) -> dict[str, float]:
    order_scores = []
    qa_scores = []
    critical = []
    for case in cases:
        n_steps = max(len(case["expected_severities"]) - 1, 1)
        order_scores.append(
            0.5 * float(case["monotonic_fraction"])
            + 0.3 * ((float(case["spearman_rho"]) + 1.0) / 2.0)
            + 0.2 * (1.0 - min(float(case["violation_count"]) / n_steps, 1.0))
        )
        span_ok = float(case["expected_span"] >= 0.5)
        drift_ok = float(case["max_unmasked_delta"] <= 0.01)
        mask_ok = float(0.2 <= case["mask_area_fraction"] <= 0.75)
        scalp_ok = float(case["scalp_localization_valid"])
        qa_scores.append(float(np.mean([float(case["auto_pass"]), span_ok, drift_ok, mask_ok, scalp_ok])))
        critical.append(float(case["critical_error"]))
    return {
        "s_order": float(np.mean(order_scores)) if order_scores else 0.0,
        "s_qa": float(np.mean(qa_scores)) if qa_scores else 0.0,
        "e_crit": float(np.mean(critical)) if critical else 1.0,
    }


def _fragility(components: dict[str, float], weights: tuple[float, float, float]) -> float:
    alpha, beta, gamma = weights
    return float(alpha * (1.0 - components["s_order"]) + beta * (1.0 - components["s_qa"]) + gamma * components["e_crit"])


def _evaluate_rendered_pipeline(
    pipeline_id: str,
    split: str,
    perturbation: dict[str, Any],
    eval_rows: list[dict[str, str]],
    base_rows_by_case: dict[str, dict[str, str]],
    vectors: np.ndarray,
    scorer: torch.nn.Module,
    labels: list[int],
    image_size: int,
    device: torch.device,
    train_std: np.ndarray,
    output_dir: Path,
    seed: int,
    make_contact: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[tuple[dict[str, str], np.ndarray]]] = {}
    perturbed_vectors = _perturb_vectors(vectors, perturbation, image_size, train_std, rng)
    for row, vector in zip(eval_rows, perturbed_vectors):
        grouped.setdefault(row["case_id"], []).append((row, vector))
    cases: list[dict[str, Any]] = []
    contact_rows: list[Image.Image] = []
    resize = Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
    with torch.no_grad():
        for case_id, items in sorted(grouped.items()):
            items = sorted(items, key=lambda item: float(item[0]["target_severity"]))
            base = base_rows_by_case[case_id]
            source_image = _perturb_image(_load_image(base, image_size), perturbation, rng)
            source = ToTensor()(source_image).unsqueeze(0).to(device)
            source_score = _score_image(scorer, source, labels)
            panels = [_add_label(_to_pil(source[0]), [f"{pipeline_id}", f"source {base['source_severity']}", f"exp {source_score['expected_severity']:.2f}"])]
            target_summaries: list[dict[str, Any]] = []
            for row, vector in items:
                edited, mask = _render(source, source_image, vector, device)
                score = _score_image(scorer, edited, labels)
                visual = _visual_delta_metrics(source, edited, mask)
                target_summaries.append({"target": float(row["target_severity"]), **score, **visual})
                panels.append(
                    _add_label(
                        _to_pil(edited[0]),
                        [
                            f"target {float(row['target_severity']):.2f}",
                            f"exp {score['expected_severity']:.2f}",
                            f"unmask {visual['unmasked_mean_absolute_delta']:.4f}",
                        ],
                    )
                )
            expected = [item["expected_severity"] for item in target_summaries]
            sev = _case_metrics(expected)
            max_drift = float(max(item["unmasked_mean_absolute_delta"] for item in target_summaries))
            mask_area = float(target_summaries[0]["mask_area_fraction"])
            scalp_valid = max_drift <= 0.01 and 0.2 <= mask_area <= 0.75
            auto_pass = bool(sev["monotonic_fraction"] >= 0.8 and sev["expected_span"] >= 0.5 and max_drift <= 0.01 and 0.2 <= mask_area <= 0.75)
            critical_error = bool((not scalp_valid) or mask_area < 0.15 or mask_area > 0.85)
            cases.append(
                {
                    "case_id": case_id,
                    "expected_severities": expected,
                    **sev,
                    "max_unmasked_delta": max_drift,
                    "mask_area_fraction": mask_area,
                    "scalp_localization_valid": scalp_valid,
                    "auto_pass": auto_pass,
                    "critical_error": critical_error,
                }
            )
            if make_contact:
                contact_rows.append(_compose_grid(panels, columns=len(panels)))
    components = _score_components(cases)
    result = {
        "pipeline_id": pipeline_id,
        "split": split,
        "perturbation": perturbation["id"],
        "perturbation_family": perturbation["family"],
        "perturbation_level": perturbation["level"],
        "case_count": len(cases),
        "auto_pass_rate": float(np.mean([c["auto_pass"] for c in cases])),
        "mean_monotonic_fraction": float(np.mean([c["monotonic_fraction"] for c in cases])),
        "mean_violation_count": float(np.mean([c["violation_count"] for c in cases])),
        "mean_spearman_rho": float(np.mean([c["spearman_rho"] for c in cases])),
        "mean_expected_span": float(np.mean([c["expected_span"] for c in cases])),
        "mean_max_unmasked_delta": float(np.mean([c["max_unmasked_delta"] for c in cases])),
        "mean_mask_area_fraction": float(np.mean([c["mask_area_fraction"] for c in cases])),
        **components,
        "fragility_balanced": _fragility(components, WEIGHT_SETTINGS["balanced"]),
        "cases": cases,
    }
    if make_contact and contact_rows:
        path = output_dir / f"{split}_{pipeline_id}_{perturbation['id']}_contact_sheet.png"
        _compose_grid(contact_rows, columns=1).save(path)
        result["contact_sheet"] = str(path)
    return result


def _archived_diffusion_result(split: str, perturbation: dict[str, Any], exp06: dict[str, Any]) -> dict[str, Any]:
    sweep = exp06[f"{split}_sweep"]
    mono = float(sweep["mean_generated_monotonic_fraction"])
    auto = float(sweep["generated_auto_pass_rate"])
    span = float(sweep["mean_generated_expected_span"])
    drift = float(sweep["mean_generated_max_unmasked_delta"])
    # Archived contact sheets show non-scalp hallucination; use a conservative
    # critical-error proxy because per-perturbation diffusion images are absent.
    ecrit = 1.0 if split == "test" else 0.5
    rho_proxy = max(-1.0, min(1.0, 2.0 * mono - 1.0))
    s_order = 0.5 * mono + 0.3 * ((rho_proxy + 1.0) / 2.0) + 0.2 * mono
    s_qa = float(np.mean([auto, float(span >= 0.5), float(drift <= 0.01), 1.0, float(drift <= 0.01)]))
    components = {"s_order": s_order, "s_qa": s_qa, "e_crit": ecrit}
    return {
        "pipeline_id": "P2_diffusion_archived_exp06",
        "split": split,
        "perturbation": perturbation["id"],
        "perturbation_family": perturbation["family"],
        "perturbation_level": perturbation["level"],
        "case_count": int(sweep["case_count"]),
        "auto_pass_rate": auto,
        "mean_monotonic_fraction": mono,
        "mean_violation_count": None,
        "mean_spearman_rho": rho_proxy,
        "mean_expected_span": span,
        "mean_max_unmasked_delta": drift,
        "mean_mask_area_fraction": None,
        **components,
        "fragility_balanced": _fragility(components, WEIGHT_SETTINGS["balanced"]),
        "archived_comparator": True,
    }


def _perturbation_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"id": "nominal", "family": "none", "level": "nominal", "value": 0.0}]
    for level, px, tn, bc, gn, pn in [("low", 1, 0.03, 0.05, 0.01, 0.5), ("high", 3, 0.08, 0.15, 0.03, 1.0)]:
        specs.extend(
            [
                {"id": f"mask_erode_{level}", "family": "mask_erode", "level": level, "px": px},
                {"id": f"mask_dilate_{level}", "family": "mask_dilate", "level": level, "px": px},
                {"id": f"mask_shift_pos_{level}", "family": "mask_shift", "level": level, "px": px, "dx": 1.0, "dy": 1.0},
                {"id": f"mask_shift_neg_{level}", "family": "mask_shift", "level": level, "px": px, "dx": -1.0, "dy": -1.0},
                {"id": f"target_noise_{level}", "family": "target_noise", "level": level, "value": tn},
                {"id": f"brightness_up_{level}", "family": "brightness", "level": level, "value": bc},
                {"id": f"brightness_down_{level}", "family": "brightness", "level": level, "value": -bc},
                {"id": f"contrast_up_{level}", "family": "contrast", "level": level, "value": bc},
                {"id": f"contrast_down_{level}", "family": "contrast", "level": level, "value": -bc},
                {"id": f"image_noise_{level}", "family": "image_noise", "level": level, "value": gn},
                {"id": f"parameter_noise_{level}", "family": "parameter_noise", "level": level, "value": pn},
            ]
        )
    return specs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys and key != "cases":
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _summaries(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault((row["pipeline_id"], row["perturbation_family"]), []).append(row)
    mean_fragility = []
    components = []
    for (pipeline, family), items in sorted(grouped.items()):
        mean_fragility.append(
            {
                "pipeline_id": pipeline,
                "perturbation_family": family,
                "mean_fragility_balanced": float(np.mean([item["fragility_balanced"] for item in items])),
                "mean_auto_pass_rate": float(np.mean([item["auto_pass_rate"] for item in items])),
                "mean_monotonic_fraction": float(np.mean([item["mean_monotonic_fraction"] for item in items])),
                "mean_critical_error_rate": float(np.mean([item["e_crit"] for item in items])),
            }
        )
        components.append(
            {
                "pipeline_id": pipeline,
                "perturbation_family": family,
                "s_order": float(np.mean([item["s_order"] for item in items])),
                "s_qa": float(np.mean([item["s_qa"] for item in items])),
                "e_crit": float(np.mean([item["e_crit"] for item in items])),
                "ordering_only": float(np.mean([1.0 - item["s_order"] for item in items])),
                "qa_only": float(np.mean([1.0 - item["s_qa"] for item in items])),
                "critical_only": float(np.mean([item["e_crit"] for item in items])),
            }
        )
    weight_rows = []
    by_pipeline: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_pipeline.setdefault(row["pipeline_id"], []).append(row)
    for pipeline, items in sorted(by_pipeline.items()):
        for setting, weights in WEIGHT_SETTINGS.items():
            weight_rows.append(
                {
                    "pipeline_id": pipeline,
                    "weight_setting": setting,
                    "alpha": weights[0],
                    "beta": weights[1],
                    "gamma": weights[2],
                    "mean_fragility": float(np.mean([_fragility(item, weights) for item in items])),
                }
            )
    return mean_fragility, components, weight_rows


def _write_svg_bar(path: Path, rows: list[dict[str, Any]]) -> None:
    pipelines = sorted({row["pipeline_id"] for row in rows})
    vals = {p: float(np.mean([r["fragility_balanced"] for r in rows if r["pipeline_id"] == p and r["split"] == "test"])) for p in pipelines}
    width, height = 900, 460
    ml, mt, mb = 80, 50, 90
    plot_h = height - mt - mb
    max_v = max(vals.values()) if vals else 1.0
    bar_w = 110
    gap = 60
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">Experiment 09 Mean Test Fragility</text>')
    parts.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" stroke="#333"/>')
    parts.append(f'<line x1="{ml}" y1="{mt+plot_h}" x2="{width-30}" y2="{mt+plot_h}" stroke="#333"/>')
    colors = ["#376092", "#c0504d", "#70ad47", "#9e7bb5"]
    for idx, pipeline in enumerate(pipelines):
        x = ml + 45 + idx * (bar_w + gap)
        h = 0 if max_v <= 0 else vals[pipeline] / max_v * plot_h
        y = mt + plot_h - h
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{colors[idx % len(colors)]}"/>')
        parts.append(f'<text x="{x+bar_w/2}" y="{y-8}" text-anchor="middle" font-family="Arial" font-size="13">{vals[pipeline]:.3f}</text>')
        label = pipeline.replace("_", " ")
        parts.append(f'<text x="{x+bar_w/2}" y="{mt+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="11">{label}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts))


def _write_human_template(path: Path) -> None:
    rows = []
    for case_id in ["val_case_1", "val_case_2", "test_case_1", "test_case_2", "test_case_3", "test_case_4"]:
        for pipeline in ["P1_proxy", "P2_diffusion_archived_exp06", "P4_sparse4_ridge"]:
            rows.append(
                {
                    "rater_id": "",
                    "case_id": case_id,
                    "pipeline_id": pipeline,
                    "perturbation": "nominal_or_selected_stress",
                    "source_preservation_1_5": "",
                    "scalp_localisation_1_5": "",
                    "monotonic_progression_1_5": "",
                    "anatomical_plausibility_1_5": "",
                    "overall_acceptability_1_5": "",
                    "comments": "",
                }
            )
    _write_csv(path, rows)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--pair-manifest", type=Path, default=Path("experiments/results/exp02/source_aware_proxy_batch_v1/proxy_pair_manifest.csv"))
    parser.add_argument("--scorer-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/exp09/fragility_aware_evaluation_v1"))
    parser.add_argument("--exp06-summary", type=Path, default=Path("experiments/results/exp06/severity_delta_cross_attention_v1/summary.json"))
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=9009)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args.pair_manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer, labels, image_size = _load_scorer(args.scorer_checkpoint, device)
    target_labels = [label for label in TARGET_LABELS if label in labels]
    stats_full = _stats_cache_for_rows(rows, image_size)
    stats_sparse2 = _masked_stats(stats_full, SPARSE2_STATS)
    stats_sparse4 = _masked_stats(stats_full, SPARSE4_STATS)
    severities = [float(row["source_severity"]) for row in rows] + [float(row["target_severity"]) for row in rows]
    min_severity, max_severity = min(severities), max(severities)
    train_rows = [row for row in rows if row["split"] == "train"]
    dataset_vocab = sorted({row.get("dataset_key", "") for row in train_rows})
    schema_vocab = sorted({row.get("label_schema", "") for row in train_rows})
    y_train = _target_matrix(train_rows, stats_full, min_severity, max_severity)
    _, train_std = _fit_standardizer(y_train)
    sparse2_model = _fit_model(train_rows, "image_stats", stats_sparse2, min_severity, max_severity, dataset_vocab, schema_vocab, args.ridge_alpha)
    sparse4_model = _fit_model(train_rows, "image_stats", stats_sparse4, min_severity, max_severity, dataset_vocab, schema_vocab, args.ridge_alpha)
    exp06 = json.loads(args.exp06_summary.read_text())

    all_results: list[dict[str, Any]] = []
    specs = _perturbation_specs()
    for split in ["val", "test"]:
        split_specs = [spec for spec in specs if spec["level"] in {"nominal", "low"}] if split == "val" else specs
        cases = _rows_by_split(rows, split)
        base_by_case = {row["case_id"]: row for row in cases}
        base_eval_rows = _eval_rows(cases, target_labels)
        for spec in split_specs:
            rng = np.random.default_rng(args.seed + _stable_seed(args.seed, split, spec["id"]))
            perturbed_rows = [_perturb_eval_row(row, spec, rng) for row in base_eval_rows]
            # Recompute image statistics for acquisition perturbations.
            if spec["family"] in {"brightness", "contrast", "image_noise"}:
                pert_stats: dict[str, np.ndarray] = {}
                for case in cases:
                    image_rng = np.random.default_rng(args.seed + _stable_seed(args.seed, case["case_id"], spec["id"]))
                    pert_image = _perturb_image(_load_image(case, image_size), spec, image_rng)
                    pert_stats[case["case_id"]] = _image_stats_from_image(case, pert_image, image_size)
            else:
                pert_stats = stats_full
            p1_vectors = _target_matrix(perturbed_rows, pert_stats, min_severity, max_severity)
            p3_vectors = _predict(sparse2_model, perturbed_rows, _masked_stats(pert_stats, SPARSE2_STATS), min_severity, max_severity, dataset_vocab, schema_vocab)
            p4_vectors = _predict(sparse4_model, perturbed_rows, _masked_stats(pert_stats, SPARSE4_STATS), min_severity, max_severity, dataset_vocab, schema_vocab)
            make_contact = spec["id"] in {"nominal", "parameter_noise_high", "brightness_up_high"} and split == "test"
            for pipeline_id, vectors in [
                ("P1_proxy", p1_vectors),
                ("P3_sparse2_ridge", p3_vectors),
                ("P4_sparse4_ridge", p4_vectors),
            ]:
                all_results.append(
                    _evaluate_rendered_pipeline(
                        pipeline_id,
                        split,
                        spec,
                        perturbed_rows,
                        base_by_case,
                        vectors,
                        scorer,
                        labels,
                        image_size,
                        device,
                        train_std,
                        args.output_dir,
                        args.seed + len(all_results),
                        make_contact,
                    )
                )
            all_results.append(_archived_diffusion_result(split, spec, exp06))

    mean_fragility, component_rows, weight_rows = _summaries(all_results)
    _write_csv(args.output_dir / "pipeline_comparison.csv", [
        {"pipeline_id": p, "description": d}
        for p, d in [
            ("P1_proxy", "QA-gated deterministic proxy baseline"),
            ("P2_diffusion_archived_exp06", "Best archived diffusion editor from Exp06, severity-delta cross-attention"),
            ("P3_sparse2_ridge", "Sparse2 ridge controller: hair-presence index + masked grayscale mean"),
            ("P4_sparse4_ridge", "Sparse4 ridge controller: sparse2 + mask area fraction + texture p90"),
        ]
    ])
    _write_csv(args.output_dir / "perturbation_specification.csv", specs)
    _write_csv(args.output_dir / "fragility_by_pipeline_and_perturbation.csv", all_results)
    _write_csv(args.output_dir / "mean_fragility_by_pipeline_family.csv", mean_fragility)
    _write_csv(args.output_dir / "component_breakdown.csv", component_rows)
    _write_csv(args.output_dir / "weight_sensitivity.csv", weight_rows)
    _write_human_template(args.output_dir / "human_face_validity_rating_template.csv")
    _write_svg_bar(args.output_dir / "figure1_fragility_curves.svg", all_results)

    for src, name in [
        (Path("exp06_severity_delta_cross_attention_v1_samples/test_exp06_vs_proxy_contact_sheet.png"), "contact_sheet_P2_diffusion_failure.png"),
        (Path("experiments/results/exp08/qa_refined_proxy_controller_v1/test_sparse4_ridge_contact_sheet.png"), "contact_sheet_P4_sparse4_nominal_reference.png"),
    ]:
        if src.exists():
            shutil.copyfile(src, args.output_dir / name)

    summary = {
        "experiment": "exp09_fragility_aware_evaluation_v1",
        "status": "completed_computational_fragility; human_ratings_template_prepared",
        "target_labels": target_labels,
        "weights": WEIGHT_SETTINGS,
        "p2_limitation": "P2 uses archived Exp06 diffusion metrics/contact-sheet evidence rather than per-perturbation reruns.",
        "results": all_results,
        "mean_fragility_by_pipeline_family": mean_fragility,
        "component_breakdown": component_rows,
        "weight_sensitivity": weight_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
