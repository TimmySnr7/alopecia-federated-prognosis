"""Render and evaluate Experiment 07 proxy-parameter controllers.

This script is intended for AI5, where the source image paths in the
Experiment 02 proxy-pair manifest are available. It fits the lightweight
parameter controllers from Experiment 07 snapshot v1, renders predicted proxy
sweeps, and evaluates them with the same scorer/QA gate used by recent
experiments.
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
    _build_features,
    _fit_ridge,
    _fit_standardizer,
    _load_rows,
    _standardize,
    _target_vector,
)


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
            item["pair_id"] = f"{row['case_id']}__eval_target_{label}"
            item["target_severity"] = str(label)
            item["pair_type"] = "identity" if int(float(row["source_severity"])) == label else "severity_shift"
            eval_rows.append(item)
    return eval_rows


def _fit_model(
    train_rows: list[dict[str, str]],
    feature_mode: str,
    min_severity: float,
    max_severity: float,
    dataset_vocab: list[str],
    schema_vocab: list[str],
    ridge_alpha: float,
) -> dict[str, Any]:
    y_train = np.vstack([_target_vector(row, min_severity, max_severity) for row in train_rows])
    y_mean, y_std = _fit_standardizer(y_train)

    if feature_mode == "constant":
        return {
            "feature_mode": feature_mode,
            "y_mean": y_mean,
            "y_std": y_std,
        }

    x_train = _build_features(
        train_rows,
        mode=feature_mode,
        min_severity=min_severity,
        max_severity=max_severity,
        dataset_vocab=dataset_vocab,
        schema_vocab=schema_vocab,
    )
    if x_train.shape[1] > 1:
        x_mean, x_std = _fit_standardizer(x_train[:, 1:])
        x_train_scaled = np.column_stack([x_train[:, 0], _standardize(x_train[:, 1:], x_mean, x_std)])
    else:
        x_mean, x_std = np.zeros(0), np.ones(0)
        x_train_scaled = x_train

    weights = _fit_ridge(x_train_scaled, _standardize(y_train, y_mean, y_std), ridge_alpha)
    return {
        "feature_mode": feature_mode,
        "weights": weights,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def _predict(
    model: dict[str, Any],
    rows: list[dict[str, str]],
    min_severity: float,
    max_severity: float,
    dataset_vocab: list[str],
    schema_vocab: list[str],
) -> np.ndarray:
    if model["feature_mode"] == "constant":
        return np.repeat(model["y_mean"][None, :], len(rows), axis=0)

    x = _build_features(
        rows,
        mode=model["feature_mode"],
        min_severity=min_severity,
        max_severity=max_severity,
        dataset_vocab=dataset_vocab,
        schema_vocab=schema_vocab,
    )
    if x.shape[1] > 1:
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


def _clip_parameters(vector: np.ndarray) -> np.ndarray:
    clipped = np.asarray(vector, dtype=np.float64).copy()
    # Keep ellipse valid but do not force it into the QA range; the gate should
    # still expose degenerate predictions if they occur.
    clipped[0] = np.clip(clipped[0], 0.05, 0.95)
    clipped[1] = np.clip(clipped[1], 0.05, 0.95)
    clipped[2] = np.clip(clipped[2], 0.02, 0.80)
    clipped[3] = np.clip(clipped[3], 0.02, 0.80)
    clipped[4] = max(float(clipped[4]), 0.0)
    clipped[5] = max(float(clipped[5]), 0.0)
    return clipped


def _render_effective_proxy(
    source: torch.Tensor,
    resized_image: Image.Image,
    vector: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    vector = _clip_parameters(vector)
    by_name = dict(zip(PARAMETER_NAMES, [float(value) for value in vector]))
    blurred_image = resized_image.filter(ImageFilter.GaussianBlur(radius=by_name["texture_blur_radius"]))
    blurred = ToTensor()(blurred_image).unsqueeze(0).to(device)
    mask = _build_ellipse_mask(
        [
            by_name["ellipse_cx"],
            by_name["ellipse_cy"],
            by_name["ellipse_rx"],
            by_name["ellipse_ry"],
        ],
        image_size=source.shape[-1],
        device=device,
        blur_radius=by_name["mask_blur_radius"],
    )
    dark_detail = (blurred - source).clamp(min=0.0)
    edited = source.clone()
    edited = edited - by_name["lower_hair_enhancement_eff"] * dark_detail
    edited = edited + by_name["lower_tone_shift_eff"]
    edited = edited + by_name["upper_hair_suppression_eff"] * dark_detail
    edited = edited + by_name["upper_smoothing_eff"] * (blurred - edited)
    edited = edited + by_name["upper_tone_shift_eff"]
    edited = torch.clamp(edited, 0.0, 1.0)
    return torch.clamp(source * (1.0 - mask) + edited * mask, 0.0, 1.0), mask


def _compose_grid(panels: list[Image.Image], columns: int) -> Image.Image:
    width, height = panels[0].size
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (width * columns, height * rows), (255, 255, 255))
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % columns) * width, (index // columns) * height))
    return canvas


def _case_passes(
    expected_values: list[float],
    max_unmasked_delta: float,
    mask_area_fraction: float,
    min_monotonic_fraction: float,
    min_expected_span: float,
    max_unmasked_delta_allowed: float,
    mask_area_min: float,
    mask_area_max: float,
) -> bool:
    return (
        _monotonic_fraction(expected_values) >= min_monotonic_fraction
        and (max(expected_values) - min(expected_values)) >= min_expected_span
        and max_unmasked_delta <= max_unmasked_delta_allowed
        and mask_area_min <= mask_area_fraction <= mask_area_max
    )


def _evaluate_model(
    model_name: str,
    eval_rows: list[dict[str, str]],
    predictions: np.ndarray,
    scorer: torch.nn.Module,
    scorer_labels: list[int],
    image_size: int,
    device: torch.device,
    output_dir: Path,
    split: str,
    args: Any,
) -> dict[str, Any]:
    resize = Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
    grouped: dict[str, list[tuple[dict[str, str], np.ndarray]]] = {}
    for row, prediction in zip(eval_rows, predictions):
        grouped.setdefault(row["case_id"], []).append((row, prediction))

    case_summaries: list[dict[str, Any]] = []
    contact_rows: list[Image.Image] = []

    with torch.no_grad():
        for case_id, items in sorted(grouped.items()):
            items = sorted(items, key=lambda item: int(float(item[0]["target_severity"])))
            base_row = items[0][0]
            image = Image.open(base_row["image_path"]).convert("RGB")
            resized = resize(image)
            source = ToTensor()(resized).unsqueeze(0).to(device)
            source_score = _score_image(scorer, source, scorer_labels)
            panels = [
                _add_label(
                    _to_pil(source[0]),
                    [
                        f"source {base_row['source_severity']}",
                        f"pred {source_score['predicted_severity']}",
                        f"exp {source_score['expected_severity']:.2f}",
                    ],
                )
            ]

            target_summaries: list[dict[str, Any]] = []
            for row, vector in items:
                edited, mask = _render_effective_proxy(source, resized, vector, device)
                score = _score_image(scorer, edited, scorer_labels)
                metrics = _visual_delta_metrics(source, edited, mask)
                target_summary = {
                    "target_severity": int(float(row["target_severity"])),
                    **score,
                    **metrics,
                }
                target_summaries.append(target_summary)
                panels.append(
                    _add_label(
                        _to_pil(edited[0]),
                        [
                            f"target {target_summary['target_severity']}",
                            f"pred {score['predicted_severity']} ({score['predicted_confidence']:.2f})",
                            f"exp {score['expected_severity']:.2f}",
                            f"unmask {metrics['unmasked_mean_absolute_delta']:.4f}",
                        ],
                    )
                )

            expected_values = [item["expected_severity"] for item in target_summaries]
            max_unmasked = max(item["unmasked_mean_absolute_delta"] for item in target_summaries)
            mask_area = target_summaries[0]["mask_area_fraction"]
            case_summary = {
                "case_id": case_id,
                "dataset_key": base_row["dataset_key"],
                "source_severity": int(float(base_row["source_severity"])),
                "expected_severities": expected_values,
                "monotonic_fraction": _monotonic_fraction(expected_values),
                "expected_span": max(expected_values) - min(expected_values),
                "max_unmasked_mean_absolute_delta": max_unmasked,
                "mask_area_fraction": mask_area,
                "auto_pass": _case_passes(
                    expected_values,
                    max_unmasked,
                    mask_area,
                    args.min_monotonic_fraction,
                    args.min_expected_span,
                    args.max_unmasked_delta,
                    args.mask_area_min,
                    args.mask_area_max,
                ),
            }
            case_summaries.append(case_summary)
            contact_rows.append(_compose_grid(panels, columns=len(panels)))

    contact_sheet_path = output_dir / f"{split}_{model_name}_contact_sheet.png"
    if contact_rows:
        _compose_grid(contact_rows, columns=1).save(contact_sheet_path)

    return {
        "split": split,
        "model_name": model_name,
        "case_count": len(case_summaries),
        "auto_pass_count": sum(1 for item in case_summaries if item["auto_pass"]),
        "auto_pass_rate": float(np.mean([item["auto_pass"] for item in case_summaries])) if case_summaries else 0.0,
        "mean_monotonic_fraction": float(np.mean([item["monotonic_fraction"] for item in case_summaries])) if case_summaries else 0.0,
        "mean_expected_span": float(np.mean([item["expected_span"] for item in case_summaries])) if case_summaries else 0.0,
        "mean_max_unmasked_delta": float(np.mean([item["max_unmasked_mean_absolute_delta"] for item in case_summaries])) if case_summaries else 0.0,
        "mean_mask_area_fraction": float(np.mean([item["mask_area_fraction"] for item in case_summaries])) if case_summaries else 0.0,
        "contact_sheet": str(contact_sheet_path),
        "cases": case_summaries,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        default=Path("experiments/results/exp02/source_aware_proxy_batch_v1/proxy_pair_manifest.csv"),
    )
    parser.add_argument("--scorer-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/exp07/proxy_parameter_controller_render_v1"),
    )
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--min-monotonic-fraction", type=float, default=0.8)
    parser.add_argument("--min-expected-span", type=float, default=0.5)
    parser.add_argument("--max-unmasked-delta", type=float, default=0.01)
    parser.add_argument("--mask-area-min", type=float, default=0.2)
    parser.add_argument("--mask-area-max", type=float, default=0.75)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args.pair_manifest)
    severities = [float(row["source_severity"]) for row in rows] + [
        float(row["target_severity"]) for row in rows
    ]
    min_severity = min(severities)
    max_severity = max(severities)
    train_rows = [row for row in rows if row["split"] == "train"]
    dataset_vocab = sorted({row.get("dataset_key", "") for row in train_rows})
    schema_vocab = sorted({row.get("label_schema", "") for row in train_rows})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer, scorer_labels, image_size = _load_scorer(args.scorer_checkpoint, device)

    models = {
        "constant_mean": _fit_model(
            train_rows,
            "constant",
            min_severity,
            max_severity,
            dataset_vocab,
            schema_vocab,
            args.ridge_alpha,
        ),
        "severity_ridge": _fit_model(
            train_rows,
            "severity",
            min_severity,
            max_severity,
            dataset_vocab,
            schema_vocab,
            args.ridge_alpha,
        ),
        "metadata_ridge": _fit_model(
            train_rows,
            "metadata",
            min_severity,
            max_severity,
            dataset_vocab,
            schema_vocab,
            args.ridge_alpha,
        ),
    }

    summary: dict[str, Any] = {
        "experiment": "exp07_proxy_parameter_controller_render_v1",
        "pair_manifest": str(args.pair_manifest),
        "scorer_checkpoint": str(args.scorer_checkpoint),
        "device": str(device),
        "image_size": image_size,
        "scorer_labels": scorer_labels,
        "qa_thresholds": {
            "min_monotonic_fraction": args.min_monotonic_fraction,
            "min_expected_span": args.min_expected_span,
            "max_unmasked_delta": args.max_unmasked_delta,
            "mask_area_min": args.mask_area_min,
            "mask_area_max": args.mask_area_max,
        },
        "models": {},
    }

    for split in ["val", "test"]:
        split_case_rows = _case_rows(rows, split)
        eval_rows = _eval_rows_for_cases(split_case_rows, scorer_labels)
        for model_name, model in models.items():
            predictions = _predict(
                model,
                eval_rows,
                min_severity,
                max_severity,
                dataset_vocab,
                schema_vocab,
            )
            split_summary = _evaluate_model(
                model_name,
                eval_rows,
                predictions,
                scorer,
                scorer_labels,
                image_size,
                device,
                args.output_dir,
                split,
                args,
            )
            summary["models"].setdefault(model_name, {})[split] = split_summary

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
