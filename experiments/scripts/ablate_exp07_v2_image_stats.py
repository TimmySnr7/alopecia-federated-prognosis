"""Ablate image statistics for Experiment 07 v2."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.scripts.run_exp07_v2_image_aware_proxy_controller import (
    IMAGE_STAT_NAMES,
    _case_rows,
    _eval_rows_for_cases,
    _fit_model,
    _load_rows,
    _predict,
    _stats_cache,
    _target_matrix,
)
from experiments.scripts.train_exp07_proxy_parameter_controller import _fit_standardizer


def _mask_stats(
    stats_by_case: dict[str, np.ndarray],
    selected: list[int],
) -> dict[str, np.ndarray]:
    masked: dict[str, np.ndarray] = {}
    for case_id, values in stats_by_case.items():
        output = np.zeros_like(values)
        if selected:
            output[selected] = values[selected]
        masked[case_id] = output
    return masked


def _mae(y_true: np.ndarray, y_pred: np.ndarray, train_std: np.ndarray) -> dict[str, float]:
    err = np.abs(y_true - y_pred)
    return {
        "mae": float(err.mean()),
        "normalized_mae": float((err / train_std).mean()),
    }


def _evaluate_feature_set(
    rows: list[dict[str, str]],
    selected: list[int],
    scorer_labels: list[int],
    image_size: int,
    ridge_alpha: float,
) -> dict[str, Any]:
    severities = [float(row["source_severity"]) for row in rows] + [
        float(row["target_severity"]) for row in rows
    ]
    min_severity, max_severity = min(severities), max(severities)
    train_rows = [row for row in rows if row["split"] == "train"]
    dataset_vocab = sorted({row.get("dataset_key", "") for row in train_rows})
    schema_vocab = sorted({row.get("label_schema", "") for row in train_rows})
    full_stats = _stats_cache(rows, image_size)
    stats = _mask_stats(full_stats, selected)

    y_train = _target_matrix(train_rows, full_stats, min_severity, max_severity)
    _, train_std = _fit_standardizer(y_train)
    model = _fit_model(
        train_rows,
        "image_stats",
        stats,
        min_severity,
        max_severity,
        dataset_vocab,
        schema_vocab,
        ridge_alpha,
    )

    results: dict[str, Any] = {
        "selected_stats": [IMAGE_STAT_NAMES[index] for index in selected],
    }
    for split in ["val", "test"]:
        eval_rows = _eval_rows_for_cases(_case_rows(rows, split), scorer_labels)
        y_true = _target_matrix(eval_rows, full_stats, min_severity, max_severity)
        y_pred = _predict(
            model,
            eval_rows,
            stats,
            min_severity,
            max_severity,
            dataset_vocab,
            schema_vocab,
        )
        results[split] = _mae(y_true, y_pred, train_std)
    return results


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        default=Path("experiments/results/exp02/source_aware_proxy_batch_v1/proxy_pair_manifest.csv"),
    )
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=Path("experiments/results/exp07/proxy_parameter_controller_v2_image_aware/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/exp07/proxy_parameter_controller_v2_image_stat_ablation"),
    )
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    args = parser.parse_args()

    rows = _load_rows(args.pair_manifest)
    reference = json.loads(args.reference_summary.read_text())
    image_size = int(reference["image_size"])
    scorer_labels = [1, 2, 3, 4, 5, 6, 7]

    individual: list[dict[str, Any]] = []
    for index, name in enumerate(IMAGE_STAT_NAMES):
        result = _evaluate_feature_set(rows, [index], scorer_labels, image_size, args.ridge_alpha)
        result["stat"] = name
        individual.append(result)
    individual = sorted(individual, key=lambda item: item["test"]["mae"])

    selected: list[int] = []
    forward: list[dict[str, Any]] = []
    remaining = list(range(len(IMAGE_STAT_NAMES)))
    for _ in IMAGE_STAT_NAMES:
        candidates: list[tuple[float, int, dict[str, Any]]] = []
        for index in remaining:
            result = _evaluate_feature_set(
                rows,
                [*selected, index],
                scorer_labels,
                image_size,
                args.ridge_alpha,
            )
            candidates.append((result["test"]["mae"], index, result))
        _, best_index, best_result = min(candidates, key=lambda item: item[0])
        selected.append(best_index)
        remaining.remove(best_index)
        best_result["step"] = len(selected)
        forward.append(best_result)

    all_stats = _evaluate_feature_set(
        rows,
        list(range(len(IMAGE_STAT_NAMES))),
        scorer_labels,
        image_size,
        args.ridge_alpha,
    )
    no_stats = _evaluate_feature_set(rows, [], scorer_labels, image_size, args.ridge_alpha)

    output = {
        "experiment": "exp07_v2_image_stat_ablation",
        "image_stat_names": IMAGE_STAT_NAMES,
        "baseline_no_image_stats": no_stats,
        "all_image_stats": all_stats,
        "individual_ranked_by_test_mae": individual,
        "forward_selection": forward,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
