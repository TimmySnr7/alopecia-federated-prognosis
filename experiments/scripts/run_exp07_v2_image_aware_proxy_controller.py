"""Experiment 07 v2: image-aware proxy-parameter controller.

V1 showed that the original proxy parameters were almost fully determined by
severity and metadata. V2 deliberately enriches the proxy-control target with
source-image statistics so that image-aware control has something meaningful to
learn. The renderer remains deterministic and QA-gated.
"""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
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
from experiments.scripts.train_exp07_proxy_parameter_controller import (
    PARAMETER_NAMES,
    _alpha_values,
    _fit_ridge,
    _fit_standardizer,
    _load_json_cell,
    _load_rows,
    _standardize,
)


IMAGE_STAT_NAMES = [
    "masked_gray_mean",
    "masked_gray_std",
    "masked_dark_fraction",
    "masked_bright_fraction",
    "masked_texture_mean",
    "masked_texture_p90",
    "unmasked_gray_mean",
    "mask_area_fraction",
    "hair_presence_index",
]


def _rows_by_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == split]


def _case_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    seen: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["split"] == split:
            seen.setdefault(row["case_id"], row)
    return list(seen.values())


def _eval_rows_for_cases(case_rows: list[dict[str, str]], labels: list[int]) -> list[dict[str, str]]:
    eval_rows: list[dict[str, str]] = []
    for row in case_rows:
        for label in labels:
            item = dict(row)
            item["target_severity"] = str(label)
            item["pair_id"] = f"{row['case_id']}__target_{label}"
            eval_rows.append(item)
    return eval_rows


def _load_image(row: dict[str, str], image_size: int) -> Image.Image:
    image = Image.open(row["image_path"]).convert("RGB")
    return Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)(image)


def _image_stats(row: dict[str, str], image_size: int) -> np.ndarray:
    image = _load_image(row, image_size)
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=3.0)).convert("L"), dtype=np.float32) / 255.0
    texture = np.abs(gray - blurred)

    ellipse = [float(value) for value in _load_json_cell(row, "ellipse_mask")]
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    cx, cy, rx, ry = ellipse
    hard_mask = (
        ((xx / image_size - cx) / max(rx, 1e-6)) ** 2
        + ((yy / image_size - cy) / max(ry, 1e-6)) ** 2
    ) <= 1.0
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


def _stats_cache(rows: list[dict[str, str]], image_size: int) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for row in rows:
        cache.setdefault(row["case_id"], _image_stats(row, image_size))
    return cache


def _target_vector_v2(
    row: dict[str, str],
    stats: np.ndarray,
    min_severity: float,
    max_severity: float,
) -> np.ndarray:
    source = float(row["source_severity"])
    target = float(row["target_severity"])
    alpha_down, alpha_up = _alpha_values(source, target, min_severity, max_severity)
    ellipse = [float(value) for value in _load_json_cell(row, "ellipse_mask")]
    params = _load_json_cell(row, "proxy_parameters")
    stat = dict(zip(IMAGE_STAT_NAMES, [float(value) for value in stats]))

    hair_presence = stat["hair_presence_index"]
    brightness = stat["masked_gray_mean"]
    texture = stat["masked_texture_mean"]
    bright_fraction = stat["masked_bright_fraction"]

    lower_multiplier = 0.70 + 0.70 * hair_presence + 1.50 * texture
    upper_suppression_multiplier = 0.80 + 0.55 * hair_presence + 0.75 * texture
    smoothing_multiplier = 0.75 + 0.35 * bright_fraction + 0.25 * (1.0 - hair_presence)
    tone_multiplier = 0.65 + 0.55 * brightness

    return np.asarray(
        [
            *ellipse,
            float(params["mask_blur_radius"]),
            float(params["texture_blur_radius"]),
            alpha_down * float(params["hair_enhancement_strength"]) * lower_multiplier,
            -alpha_down * float(params["tone_shift_strength"]) * 0.5 * tone_multiplier,
            alpha_up * float(params["hair_suppression_strength"]) * upper_suppression_multiplier,
            alpha_up * float(params["smoothing_strength"]) * smoothing_multiplier,
            alpha_up * float(params["tone_shift_strength"]) * tone_multiplier,
        ],
        dtype=np.float64,
    )


def _target_matrix(
    rows: list[dict[str, str]],
    stats_by_case: dict[str, np.ndarray],
    min_severity: float,
    max_severity: float,
) -> np.ndarray:
    return np.vstack(
        [
            _target_vector_v2(row, stats_by_case[row["case_id"]], min_severity, max_severity)
            for row in rows
        ]
    )


def _one_hot(value: str, vocabulary: list[str]) -> list[float]:
    return [1.0 if value == item else 0.0 for item in vocabulary]


def _base_features(row: dict[str, str], min_severity: float, max_severity: float) -> list[float]:
    source = float(row["source_severity"])
    target = float(row["target_severity"])
    severity_range = max(max_severity - min_severity, 1.0)
    delta = target - source
    alpha_down, alpha_up = _alpha_values(source, target, min_severity, max_severity)
    return [
        1.0,
        (source - min_severity) / severity_range,
        (target - min_severity) / severity_range,
        delta / severity_range,
        abs(delta) / severity_range,
        alpha_down,
        alpha_up,
    ]


def _feature_matrix(
    rows: list[dict[str, str]],
    mode: str,
    stats_by_case: dict[str, np.ndarray],
    min_severity: float,
    max_severity: float,
    dataset_vocab: list[str],
    schema_vocab: list[str],
) -> np.ndarray:
    features: list[list[float]] = []
    for row in rows:
        if mode == "constant":
            vector = [1.0]
        else:
            vector = _base_features(row, min_severity, max_severity)
            if mode in {"metadata", "image_stats"}:
                vector.extend(_one_hot(row.get("dataset_key", ""), dataset_vocab))
                vector.extend(_one_hot(row.get("label_schema", ""), schema_vocab))
            if mode == "image_stats":
                stats = stats_by_case[row["case_id"]].tolist()
                vector.extend(stats)
                vector.extend([stats[-1] * vector[5], stats[-1] * vector[6]])
        features.append(vector)
    return np.asarray(features, dtype=np.float64)


def _fit_model(
    train_rows: list[dict[str, str]],
    mode: str,
    stats_by_case: dict[str, np.ndarray],
    min_severity: float,
    max_severity: float,
    dataset_vocab: list[str],
    schema_vocab: list[str],
    ridge_alpha: float,
) -> dict[str, Any]:
    y_train = _target_matrix(train_rows, stats_by_case, min_severity, max_severity)
    y_mean, y_std = _fit_standardizer(y_train)
    if mode == "constant":
        return {"mode": mode, "y_mean": y_mean, "y_std": y_std}
    x_train = _feature_matrix(
        train_rows,
        mode,
        stats_by_case,
        min_severity,
        max_severity,
        dataset_vocab,
        schema_vocab,
    )
    x_mean, x_std = _fit_standardizer(x_train[:, 1:])
    x_scaled = np.column_stack([x_train[:, 0], _standardize(x_train[:, 1:], x_mean, x_std)])
    weights = _fit_ridge(x_scaled, _standardize(y_train, y_mean, y_std), ridge_alpha)
    return {
        "mode": mode,
        "weights": weights,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def _predict(
    model: dict[str, Any],
    rows: list[dict[str, str]],
    stats_by_case: dict[str, np.ndarray],
    min_severity: float,
    max_severity: float,
    dataset_vocab: list[str],
    schema_vocab: list[str],
) -> np.ndarray:
    if model["mode"] == "constant":
        return np.repeat(model["y_mean"][None, :], len(rows), axis=0)
    x = _feature_matrix(
        rows,
        model["mode"],
        stats_by_case,
        min_severity,
        max_severity,
        dataset_vocab,
        schema_vocab,
    )
    x = np.column_stack([x[:, 0], _standardize(x[:, 1:], model["x_mean"], model["x_std"])])
    return (x @ model["weights"]) * model["y_std"] + model["y_mean"]


def _score_image(scorer: torch.nn.Module, image: torch.Tensor, labels: list[int]) -> dict[str, float | int]:
    logits = scorer(image)
    probabilities = torch.softmax(logits[0], dim=0)
    predicted_index = int(probabilities.argmax().item())
    return {
        "predicted_severity": int(labels[predicted_index]),
        "predicted_confidence": float(probabilities[predicted_index].item()),
        "expected_severity": _expected_severity(probabilities, labels),
    }


def _clip(vector: np.ndarray) -> np.ndarray:
    out = vector.copy()
    out[0] = np.clip(out[0], 0.05, 0.95)
    out[1] = np.clip(out[1], 0.05, 0.95)
    out[2] = np.clip(out[2], 0.02, 0.80)
    out[3] = np.clip(out[3], 0.02, 0.80)
    out[4] = max(float(out[4]), 0.0)
    out[5] = max(float(out[5]), 0.0)
    return out


def _render(source: torch.Tensor, resized: Image.Image, vector: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    vector = _clip(vector)
    p = dict(zip(PARAMETER_NAMES, [float(value) for value in vector]))
    blurred_image = resized.filter(ImageFilter.GaussianBlur(radius=p["texture_blur_radius"]))
    blurred = ToTensor()(blurred_image).unsqueeze(0).to(device)
    mask = _build_ellipse_mask(
        [p["ellipse_cx"], p["ellipse_cy"], p["ellipse_rx"], p["ellipse_ry"]],
        image_size=source.shape[-1],
        device=device,
        blur_radius=p["mask_blur_radius"],
    )
    dark_detail = (blurred - source).clamp(min=0.0)
    edited = source.clone()
    edited = edited - p["lower_hair_enhancement_eff"] * dark_detail
    edited = edited + p["lower_tone_shift_eff"]
    edited = edited + p["upper_hair_suppression_eff"] * dark_detail
    edited = edited + p["upper_smoothing_eff"] * (blurred - edited)
    edited = edited + p["upper_tone_shift_eff"]
    edited = torch.clamp(edited, 0.0, 1.0)
    return torch.clamp(source * (1.0 - mask) + edited * mask, 0.0, 1.0), mask


def _compose_grid(panels: list[Image.Image], columns: int) -> Image.Image:
    w, h = panels[0].size
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (w * columns, h * rows), (255, 255, 255))
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % columns) * w, (index // columns) * h))
    return canvas


def _evaluate(
    model_name: str,
    split: str,
    eval_rows: list[dict[str, str]],
    predictions: np.ndarray,
    y_true: np.ndarray,
    train_std: np.ndarray,
    scorer: torch.nn.Module,
    labels: list[int],
    image_size: int,
    device: torch.device,
    output_dir: Path,
    args: Any,
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[dict[str, str], np.ndarray]]] = {}
    for row, pred in zip(eval_rows, predictions):
        grouped.setdefault(row["case_id"], []).append((row, pred))
    resize = Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
    case_summaries: list[dict[str, Any]] = []
    contact_rows: list[Image.Image] = []
    abs_error = np.abs(predictions - y_true)

    with torch.no_grad():
        for case_id, items in sorted(grouped.items()):
            items = sorted(items, key=lambda item: int(float(item[0]["target_severity"])))
            base = items[0][0]
            resized = resize(Image.open(base["image_path"]).convert("RGB"))
            source = ToTensor()(resized).unsqueeze(0).to(device)
            source_score = _score_image(scorer, source, labels)
            panels = [
                _add_label(
                    _to_pil(source[0]),
                    [f"source {base['source_severity']}", f"exp {source_score['expected_severity']:.2f}"],
                )
            ]
            target_summaries: list[dict[str, Any]] = []
            for row, pred in items:
                edited, mask = _render(source, resized, pred, device)
                score = _score_image(scorer, edited, labels)
                metrics = _visual_delta_metrics(source, edited, mask)
                target_summaries.append(
                    {
                        "target_severity": int(float(row["target_severity"])),
                        **score,
                        **metrics,
                    }
                )
                panels.append(
                    _add_label(
                        _to_pil(edited[0]),
                        [
                            f"target {row['target_severity']}",
                            f"pred {score['predicted_severity']} ({score['predicted_confidence']:.2f})",
                            f"exp {score['expected_severity']:.2f}",
                            f"unmask {metrics['unmasked_mean_absolute_delta']:.4f}",
                        ],
                    )
                )
            expected = [item["expected_severity"] for item in target_summaries]
            max_unmasked = max(item["unmasked_mean_absolute_delta"] for item in target_summaries)
            mask_area = target_summaries[0]["mask_area_fraction"]
            auto_pass = (
                _monotonic_fraction(expected) >= args.min_monotonic_fraction
                and (max(expected) - min(expected)) >= args.min_expected_span
                and max_unmasked <= args.max_unmasked_delta
                and args.mask_area_min <= mask_area <= args.mask_area_max
            )
            case_summaries.append(
                {
                    "case_id": case_id,
                    "dataset_key": base["dataset_key"],
                    "source_severity": int(float(base["source_severity"])),
                    "expected_severities": expected,
                    "monotonic_fraction": _monotonic_fraction(expected),
                    "expected_span": max(expected) - min(expected),
                    "max_unmasked_mean_absolute_delta": max_unmasked,
                    "mask_area_fraction": mask_area,
                    "auto_pass": auto_pass,
                }
            )
            contact_rows.append(_compose_grid(panels, columns=len(panels)))

    contact_sheet = output_dir / f"{split}_{model_name}_contact_sheet.png"
    if contact_rows:
        _compose_grid(contact_rows, columns=1).save(contact_sheet)
    return {
        "split": split,
        "model_name": model_name,
        "parameter_mae": float(abs_error.mean()),
        "parameter_normalized_mae": float((abs_error / train_std).mean()),
        "per_parameter_mae": {name: float(value) for name, value in zip(PARAMETER_NAMES, abs_error.mean(axis=0))},
        "case_count": len(case_summaries),
        "auto_pass_rate": float(np.mean([item["auto_pass"] for item in case_summaries])),
        "mean_monotonic_fraction": float(np.mean([item["monotonic_fraction"] for item in case_summaries])),
        "mean_expected_span": float(np.mean([item["expected_span"] for item in case_summaries])),
        "mean_max_unmasked_delta": float(np.mean([item["max_unmasked_mean_absolute_delta"] for item in case_summaries])),
        "mean_mask_area_fraction": float(np.mean([item["mask_area_fraction"] for item in case_summaries])),
        "contact_sheet": str(contact_sheet),
        "cases": case_summaries,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--pair-manifest", type=Path, default=Path("experiments/results/exp02/source_aware_proxy_batch_v1/proxy_pair_manifest.csv"))
    parser.add_argument("--scorer-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/exp07/proxy_parameter_controller_v2_image_aware"))
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--min-monotonic-fraction", type=float, default=0.8)
    parser.add_argument("--min-expected-span", type=float, default=0.5)
    parser.add_argument("--max-unmasked-delta", type=float, default=0.01)
    parser.add_argument("--mask-area-min", type=float, default=0.2)
    parser.add_argument("--mask-area-max", type=float, default=0.75)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args.pair_manifest)
    severities = [float(row["source_severity"]) for row in rows] + [float(row["target_severity"]) for row in rows]
    min_severity, max_severity = min(severities), max(severities)
    train_rows = _rows_by_split(rows, "train")
    dataset_vocab = sorted({row.get("dataset_key", "") for row in train_rows})
    schema_vocab = sorted({row.get("label_schema", "") for row in train_rows})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer, labels, image_size = _load_scorer(args.scorer_checkpoint, device)
    stats_by_case = _stats_cache(rows, image_size)
    y_train = _target_matrix(train_rows, stats_by_case, min_severity, max_severity)
    _, train_std = _fit_standardizer(y_train)

    models = {
        "constant_mean": _fit_model(train_rows, "constant", stats_by_case, min_severity, max_severity, dataset_vocab, schema_vocab, args.ridge_alpha),
        "severity_ridge": _fit_model(train_rows, "severity", stats_by_case, min_severity, max_severity, dataset_vocab, schema_vocab, args.ridge_alpha),
        "metadata_ridge": _fit_model(train_rows, "metadata", stats_by_case, min_severity, max_severity, dataset_vocab, schema_vocab, args.ridge_alpha),
        "image_stats_ridge": _fit_model(train_rows, "image_stats", stats_by_case, min_severity, max_severity, dataset_vocab, schema_vocab, args.ridge_alpha),
    }
    summary: dict[str, Any] = {
        "experiment": "exp07_v2_image_aware_proxy_parameter_controller",
        "target_definition": "image-statistic-enriched effective proxy-control vector",
        "image_stat_names": IMAGE_STAT_NAMES,
        "parameter_names": PARAMETER_NAMES,
        "scorer_checkpoint": str(args.scorer_checkpoint),
        "device": str(device),
        "image_size": image_size,
        "models": {},
    }

    for split in ["val", "test"]:
        split_case_rows = _case_rows(rows, split)
        eval_rows = _eval_rows_for_cases(split_case_rows, labels)
        y_true = _target_matrix(eval_rows, stats_by_case, min_severity, max_severity)
        for model_name, model in models.items():
            pred = _predict(model, eval_rows, stats_by_case, min_severity, max_severity, dataset_vocab, schema_vocab)
            split_summary = _evaluate(
                model_name,
                split,
                eval_rows,
                pred,
                y_true,
                train_std,
                scorer,
                labels,
                image_size,
                device,
                args.output_dir,
                args,
            )
            summary["models"].setdefault(model_name, {})[split] = split_summary

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
