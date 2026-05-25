"""Experiment 08: QA-refined proxy-parameter controller.

This experiment starts from Experiment 07's sparse image-statistic controller
and tests whether bounded post-prediction refinement can preserve parameter
accuracy while enforcing exact severity monotonicity and low drift.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode, Resize, ToTensor

from experiments.scripts.run_exp01_plausible_proxy_sweep import (
    _add_label,
    _expected_severity,
    _monotonic_fraction,
    _to_pil,
)
from experiments.scripts.run_exp01_proxy_batch_eval import _load_scorer, _visual_delta_metrics
from experiments.scripts.run_exp07_v2_image_aware_proxy_controller import (
    IMAGE_STAT_NAMES,
    PARAMETER_NAMES,
    _case_rows,
    _clip,
    _compose_grid,
    _eval_rows_for_cases,
    _fit_model,
    _load_rows,
    _predict,
    _render,
    _score_image,
    _stats_cache,
    _target_matrix,
)
from experiments.scripts.train_exp07_proxy_parameter_controller import _fit_standardizer


SPARSE2_STATS = ["hair_presence_index", "masked_gray_mean"]
SPARSE4_STATS = ["hair_presence_index", "masked_gray_mean", "mask_area_fraction", "masked_texture_p90"]


def _masked_stats(stats_by_case: dict[str, np.ndarray], selected_names: list[str]) -> dict[str, np.ndarray]:
    selected = [IMAGE_STAT_NAMES.index(name) for name in selected_names]
    out: dict[str, np.ndarray] = {}
    for case_id, values in stats_by_case.items():
        masked = np.zeros_like(values)
        masked[selected] = values[selected]
        out[case_id] = masked
    return out


def _severity_metrics(expected: list[float]) -> dict[str, float | int]:
    diffs = [b - a for a, b in zip(expected, expected[1:])]
    violations = sum(1 for value in diffs if value < 0)
    # Spearman against target order 1..n, with average-rank tie handling.
    values = np.asarray(expected, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    target = np.arange(len(values), dtype=np.float64)
    if np.std(ranks) < 1e-8:
        rho = 0.0
    else:
        rho = float(np.corrcoef(target, ranks)[0, 1])
    return {
        "monotonic_fraction": float(_monotonic_fraction(expected)),
        "monotonicity_violation_count": int(violations),
        "spearman_rho": rho,
        "expected_span": float(max(expected) - min(expected)) if expected else 0.0,
    }


def _load_source(row: dict[str, str], image_size: int, device: torch.device) -> tuple[Image.Image, torch.Tensor]:
    resized = Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)(
        Image.open(row["image_path"]).convert("RGB")
    )
    return resized, ToTensor()(resized).unsqueeze(0).to(device)


def _evaluate_case(
    base_row: dict[str, str],
    vectors: np.ndarray,
    proxy_vectors: np.ndarray,
    labels: list[int],
    scorer: torch.nn.Module,
    image_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[Image.Image], list[torch.Tensor], list[torch.Tensor]]:
    resized, source = _load_source(base_row, image_size, device)
    source_score = _score_image(scorer, source, labels)
    panels = [
        _add_label(
            _to_pil(source[0]),
            [f"source {base_row['source_severity']}", f"exp {source_score['expected_severity']:.2f}"],
        )
    ]
    target_summaries: list[dict[str, Any]] = []
    edited_images: list[torch.Tensor] = []
    proxy_images: list[torch.Tensor] = []
    for target_severity, vector, proxy_vector in zip(labels, vectors, proxy_vectors):
        edited, mask = _render(source, resized, vector, device)
        proxy, _ = _render(source, resized, proxy_vector, device)
        score = _score_image(scorer, edited, labels)
        metrics = _visual_delta_metrics(source, edited, mask)
        target_summaries.append(
            {
                "target_severity": int(target_severity),
                **score,
                **metrics,
                "proxy_l1": float((edited - proxy).abs().mean().item()),
            }
        )
        edited_images.append(edited)
        proxy_images.append(proxy)
        panels.append(
            _add_label(
                _to_pil(edited[0]),
                [
                    f"target {target_severity}",
                    f"pred {score['predicted_severity']} ({score['predicted_confidence']:.2f})",
                    f"exp {score['expected_severity']:.2f}",
                    f"unmask {metrics['unmasked_mean_absolute_delta']:.4f}",
                ],
            )
        )
    expected = [item["expected_severity"] for item in target_summaries]
    sev = _severity_metrics(expected)
    case = {
        "case_id": base_row["case_id"],
        "dataset_key": base_row["dataset_key"],
        "source_severity": int(float(base_row["source_severity"])),
        "expected_severities": expected,
        **sev,
        "max_unmasked_mean_absolute_delta": float(
            max(item["unmasked_mean_absolute_delta"] for item in target_summaries)
        ),
        "mean_proxy_l1": float(np.mean([item["proxy_l1"] for item in target_summaries])),
        "mask_area_fraction": float(target_summaries[0]["mask_area_fraction"]),
        "auto_pass": bool(
            sev["monotonic_fraction"] >= 0.8
            and sev["expected_span"] >= 0.5
            and max(item["unmasked_mean_absolute_delta"] for item in target_summaries) <= 0.01
            and 0.2 <= target_summaries[0]["mask_area_fraction"] <= 0.75
        ),
    }
    return case, panels, edited_images, proxy_images


def _case_objective(
    case: dict[str, Any],
    vectors: np.ndarray,
    init_vectors: np.ndarray,
    lambda_mono: float,
    drift_weight: float,
    movement_weight: float,
) -> float:
    violation_penalty = float(case["monotonicity_violation_count"])
    span_penalty = max(0.0, 0.5 - float(case["expected_span"]))
    drift_penalty = max(0.0, float(case["max_unmasked_mean_absolute_delta"]) - 0.01) * 100.0
    movement = float(np.sqrt(((vectors - init_vectors) ** 2).sum(axis=1)).mean())
    return (
        lambda_mono * violation_penalty
        + span_penalty
        + drift_weight * drift_penalty
        + movement_weight * movement
    )


def _refine_case(
    base_row: dict[str, str],
    init_vectors: np.ndarray,
    proxy_vectors: np.ndarray,
    labels: list[int],
    scorer: torch.nn.Module,
    image_size: int,
    device: torch.device,
    lambda_mono: float,
    seed: int,
    iterations: int,
    max_l2: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    current = init_vectors.copy()
    current_case, _, _, _ = _evaluate_case(base_row, current, proxy_vectors, labels, scorer, image_size, device)
    current_score = _case_objective(current_case, current, init_vectors, lambda_mono, 1.0, 0.2)
    best = current.copy()
    best_score = current_score
    scales = np.asarray([0, 0, 0, 0, 0, 0, 0.035, 0.0005, 0.035, 0.012, 0.0007])
    editable = np.asarray([6, 7, 8, 9, 10])
    for step in range(iterations):
        candidate = best.copy()
        target_index = int(rng.integers(0, len(labels)))
        perturb = rng.normal(0.0, scales)
        candidate[target_index, editable] += perturb[editable]
        delta = candidate - init_vectors
        norms = np.sqrt((delta * delta).sum(axis=1))
        for idx, norm in enumerate(norms):
            if norm > max_l2:
                candidate[idx] = init_vectors[idx] + delta[idx] * (max_l2 / max(norm, 1e-8))
        candidate = np.vstack([_clip(row) for row in candidate])
        cand_case, _, _, _ = _evaluate_case(
            base_row, candidate, proxy_vectors, labels, scorer, image_size, device
        )
        cand_score = _case_objective(cand_case, candidate, init_vectors, lambda_mono, 1.0, 0.2)
        temperature = max(0.005, 0.05 * (1.0 - step / max(iterations, 1)))
        if cand_score < best_score or rng.random() < np.exp((best_score - cand_score) / temperature):
            best = candidate
            best_score = cand_score
    return best


def _summarize_cases(
    model_name: str,
    split: str,
    cases: list[dict[str, Any]],
    contact_rows: list[Image.Image],
    output_dir: Path,
) -> dict[str, Any]:
    contact_sheet = output_dir / f"{split}_{model_name}_contact_sheet.png"
    if contact_rows:
        _compose_grid(contact_rows, columns=1).save(contact_sheet)
    return {
        "split": split,
        "model_name": model_name,
        "case_count": len(cases),
        "auto_pass_rate": float(np.mean([item["auto_pass"] for item in cases])),
        "mean_monotonic_fraction": float(np.mean([item["monotonic_fraction"] for item in cases])),
        "mean_violation_count": float(np.mean([item["monotonicity_violation_count"] for item in cases])),
        "mean_spearman_rho": float(np.mean([item["spearman_rho"] for item in cases])),
        "mean_expected_span": float(np.mean([item["expected_span"] for item in cases])),
        "mean_max_unmasked_delta": float(np.mean([item["max_unmasked_mean_absolute_delta"] for item in cases])),
        "mean_proxy_l1": float(np.mean([item["mean_proxy_l1"] for item in cases])),
        "mean_mask_area_fraction": float(np.mean([item["mask_area_fraction"] for item in cases])),
        "contact_sheet": str(contact_sheet),
        "cases": cases,
    }


def _evaluate_model(
    model_name: str,
    split: str,
    eval_rows: list[dict[str, str]],
    predictions: np.ndarray,
    proxy_vectors: np.ndarray,
    labels: list[int],
    scorer: torch.nn.Module,
    image_size: int,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[dict[str, str], np.ndarray, np.ndarray]]] = {}
    for row, pred, proxy in zip(eval_rows, predictions, proxy_vectors):
        grouped.setdefault(row["case_id"], []).append((row, pred, proxy))
    cases: list[dict[str, Any]] = []
    contact_rows: list[Image.Image] = []
    for _, items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: int(float(item[0]["target_severity"])))
        base = items[0][0]
        vectors = np.vstack([item[1] for item in items])
        proxy = np.vstack([item[2] for item in items])
        case, panels, _, _ = _evaluate_case(base, vectors, proxy, labels, scorer, image_size, device)
        cases.append(case)
        contact_rows.append(_compose_grid(panels, columns=len(panels)))
    return _summarize_cases(model_name, split, cases, contact_rows, output_dir)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--pair-manifest", type=Path, default=Path("experiments/results/exp02/source_aware_proxy_batch_v1/proxy_pair_manifest.csv"))
    parser.add_argument("--scorer-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/exp08/qa_refined_proxy_controller_v1"))
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--max-l2", type=float, default=0.05)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args.pair_manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer, labels, image_size = _load_scorer(args.scorer_checkpoint, device)
    stats_full = _stats_cache(rows, image_size)
    stats_sparse2 = _masked_stats(stats_full, SPARSE2_STATS)
    stats_sparse4 = _masked_stats(stats_full, SPARSE4_STATS)
    severities = [float(row["source_severity"]) for row in rows] + [float(row["target_severity"]) for row in rows]
    min_severity, max_severity = min(severities), max(severities)
    train_rows = [row for row in rows if row["split"] == "train"]
    dataset_vocab = sorted({row.get("dataset_key", "") for row in train_rows})
    schema_vocab = sorted({row.get("label_schema", "") for row in train_rows})

    sparse2_model = _fit_model(train_rows, "image_stats", stats_sparse2, min_severity, max_severity, dataset_vocab, schema_vocab, args.ridge_alpha)
    sparse4_model = _fit_model(train_rows, "image_stats", stats_sparse4, min_severity, max_severity, dataset_vocab, schema_vocab, args.ridge_alpha)

    summary: dict[str, Any] = {
        "experiment": "exp08_qa_refined_proxy_controller_v1",
        "sparse2_stats": SPARSE2_STATS,
        "sparse4_stats": SPARSE4_STATS,
        "scorer_checkpoint": str(args.scorer_checkpoint),
        "image_size": image_size,
        "device": str(device),
        "models": {},
    }

    for split in ["val", "test"]:
        eval_rows = _eval_rows_for_cases(_case_rows(rows, split), labels)
        proxy_vectors = _target_matrix(eval_rows, stats_full, min_severity, max_severity)
        sparse2_pred = _predict(sparse2_model, eval_rows, stats_sparse2, min_severity, max_severity, dataset_vocab, schema_vocab)
        sparse4_pred = _predict(sparse4_model, eval_rows, stats_sparse4, min_severity, max_severity, dataset_vocab, schema_vocab)
        preds = {
            "sparse2_ridge": sparse2_pred,
            "sparse4_ridge": sparse4_pred,
        }
        for model_name, pred in preds.items():
            summary["models"].setdefault(model_name, {})[split] = _evaluate_model(
                model_name, split, eval_rows, pred, proxy_vectors, labels, scorer, image_size, device, args.output_dir
            )

        # Refinement starts from sparse4 because it is the best normalized subset
        # in Exp07 v2 while still retaining interpretable cues.
        for lambda_mono in [0.5, 1.0, 5.0, 10.0]:
            grouped: dict[str, list[tuple[dict[str, str], np.ndarray, np.ndarray]]] = {}
            for row, pred, proxy in zip(eval_rows, sparse4_pred, proxy_vectors):
                grouped.setdefault(row["case_id"], []).append((row, pred, proxy))
            refined_cases: list[dict[str, Any]] = []
            refined_contacts: list[Image.Image] = []
            drifts: list[float] = []
            for case_offset, (_, items) in enumerate(sorted(grouped.items())):
                items = sorted(items, key=lambda item: int(float(item[0]["target_severity"])))
                base = items[0][0]
                init = np.vstack([item[1] for item in items])
                proxy = np.vstack([item[2] for item in items])
                refined = _refine_case(
                    base,
                    init,
                    proxy,
                    labels,
                    scorer,
                    image_size,
                    device,
                    lambda_mono,
                    args.seed + 1000 * case_offset + int(lambda_mono * 10),
                    args.iterations,
                    args.max_l2,
                )
                case, panels, _, _ = _evaluate_case(base, refined, proxy, labels, scorer, image_size, device)
                refined_cases.append(case)
                refined_contacts.append(_compose_grid(panels, columns=len(panels)))
                drifts.append(float(np.sqrt(((refined - init) ** 2).sum(axis=1)).mean()))
            name = f"sparse4_refined_lambda_{str(lambda_mono).replace('.', '_')}"
            result = _summarize_cases(name, split, refined_cases, refined_contacts, args.output_dir)
            result["mean_parameter_drift_l2"] = float(np.mean(drifts)) if drifts else 0.0
            result["refinement_success_rate"] = float(np.mean([item["monotonicity_violation_count"] == 0 and item["auto_pass"] for item in refined_cases])) if refined_cases else 0.0
            summary["models"].setdefault(name, {})[split] = result

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
