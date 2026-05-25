"""Diagnostics for Experiment 07 proxy-parameter controller results."""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from experiments.scripts.train_exp07_proxy_parameter_controller import (
    PARAMETER_NAMES,
    _build_features,
    _fit_ridge,
    _fit_standardizer,
    _load_rows,
    _standardize,
    _target_vector,
)


MODEL_FEATURE_MODES = {
    "constant_mean": "constant",
    "severity_ridge": "severity",
    "metadata_ridge": "metadata",
}


def _feature_names(mode: str, dataset_vocab: list[str], schema_vocab: list[str]) -> list[str]:
    if mode == "constant":
        return ["intercept"]
    base = [
        "intercept",
        "source_severity_norm",
        "target_severity_norm",
        "severity_delta_norm",
        "abs_severity_delta_norm",
        "alpha_down",
        "alpha_up",
    ]
    if mode == "metadata":
        base.extend([f"dataset_key={value}" for value in dataset_vocab])
        base.extend([f"label_schema={value}" for value in schema_vocab])
    return base


def _rows_by_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == split]


def _target_matrix(rows: list[dict[str, str]], min_severity: float, max_severity: float) -> np.ndarray:
    return np.vstack([_target_vector(row, min_severity, max_severity) for row in rows])


def _fit_predict(
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    mode: str,
    min_severity: float,
    max_severity: float,
    dataset_vocab: list[str],
    schema_vocab: list[str],
    ridge_alpha: float,
    shuffled_train_targets: np.ndarray | None = None,
) -> np.ndarray:
    y_train = (
        shuffled_train_targets
        if shuffled_train_targets is not None
        else _target_matrix(train_rows, min_severity, max_severity)
    )
    if mode == "constant":
        return np.repeat(y_train.mean(axis=0, keepdims=True), len(eval_rows), axis=0)

    y_mean, y_std = _fit_standardizer(y_train)
    x_train = _build_features(
        train_rows,
        mode=mode,
        min_severity=min_severity,
        max_severity=max_severity,
        dataset_vocab=dataset_vocab,
        schema_vocab=schema_vocab,
    )
    x_eval = _build_features(
        eval_rows,
        mode=mode,
        min_severity=min_severity,
        max_severity=max_severity,
        dataset_vocab=dataset_vocab,
        schema_vocab=schema_vocab,
    )
    if x_train.shape[1] > 1:
        x_mean, x_std = _fit_standardizer(x_train[:, 1:])
        x_train = np.column_stack([x_train[:, 0], _standardize(x_train[:, 1:], x_mean, x_std)])
        x_eval = np.column_stack([x_eval[:, 0], _standardize(x_eval[:, 1:], x_mean, x_std)])

    weights = _fit_ridge(x_train, _standardize(y_train, y_mean, y_std), ridge_alpha)
    return (x_eval @ weights) * y_std + y_mean


def _error_summary(y_true: np.ndarray, y_pred: np.ndarray, train_std: np.ndarray) -> dict[str, Any]:
    abs_error = np.abs(y_true - y_pred)
    return {
        "mae": float(abs_error.mean()),
        "normalized_mae_by_train_std": float((abs_error / train_std).mean()),
        "per_parameter_mae": {
            name: float(value) for name, value in zip(PARAMETER_NAMES, abs_error.mean(axis=0))
        },
        "per_parameter_normalized_mae": {
            name: float(value) for name, value in zip(PARAMETER_NAMES, (abs_error / train_std).mean(axis=0))
        },
    }


def _parameter_stats(values: np.ndarray) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for index, name in enumerate(PARAMETER_NAMES):
        column = values[:, index]
        stats[name] = {
            "min": float(column.min()),
            "max": float(column.max()),
            "range": float(column.max() - column.min()),
            "mean": float(column.mean()),
            "std": float(column.std()),
            "unique_count": int(len(set(np.round(column, 10).tolist()))),
        }
    return stats


def _draw_prediction_plot(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cell_w = 260
    cell_h = 230
    cols = 4
    rows = 3
    image = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(image)
    for index, name in enumerate(PARAMETER_NAMES):
        col = index % cols
        row = index // cols
        x0 = col * cell_w + 45
        y0 = row * cell_h + 35
        x1 = col * cell_w + cell_w - 25
        y1 = row * cell_h + cell_h - 40
        true_col = y_true[:, index]
        pred_col = y_pred[:, index]
        lo = float(min(true_col.min(), pred_col.min()))
        hi = float(max(true_col.max(), pred_col.max()))
        if abs(hi - lo) < 1e-10:
            lo -= 0.5
            hi += 0.5

        def sx(value: float) -> int:
            return int(x0 + (value - lo) / (hi - lo) * (x1 - x0))

        def sy(value: float) -> int:
            return int(y1 - (value - lo) / (hi - lo) * (y1 - y0))

        draw.rectangle((x0, y0, x1, y1), outline=(160, 160, 160))
        draw.line((x0, y1, x1, y0), fill=(210, 0, 0), width=2)
        for truth, pred in zip(true_col, pred_col):
            x = sx(float(truth))
            y = sy(float(pred))
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(20, 80, 180))
        draw.text((x0, y0 - 24), name, fill=(0, 0, 0))
        draw.text((x0, y1 + 6), f"min {lo:.3g} max {hi:.3g}", fill=(60, 60, 60))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        default=Path("experiments/results/exp02/source_aware_proxy_batch_v1/proxy_pair_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/exp07/proxy_parameter_controller_diagnostics_v1"),
    )
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = _load_rows(args.pair_manifest)
    severities = [float(row["source_severity"]) for row in rows] + [
        float(row["target_severity"]) for row in rows
    ]
    min_severity = min(severities)
    max_severity = max(severities)
    train_rows = _rows_by_split(rows, "train")
    val_rows = _rows_by_split(rows, "val")
    test_rows = _rows_by_split(rows, "test")
    dataset_vocab = sorted({row.get("dataset_key", "") for row in train_rows})
    schema_vocab = sorted({row.get("label_schema", "") for row in train_rows})

    y_train = _target_matrix(train_rows, min_severity, max_severity)
    y_val = _target_matrix(val_rows, min_severity, max_severity)
    y_test = _target_matrix(test_rows, min_severity, max_severity)
    _, train_std = _fit_standardizer(y_train)
    y_all = _target_matrix(rows, min_severity, max_severity)

    rng = np.random.default_rng(args.seed)
    shuffled_train_targets = y_train.copy()
    rng.shuffle(shuffled_train_targets, axis=0)

    diagnostics: dict[str, Any] = {
        "experiment": "exp07_proxy_parameter_controller_diagnostics_v1",
        "pair_manifest": str(args.pair_manifest),
        "parameter_names": PARAMETER_NAMES,
        "split_case_ids": {
            "train": sorted({row["case_id"] for row in train_rows}),
            "val": sorted({row["case_id"] for row in val_rows}),
            "test": sorted({row["case_id"] for row in test_rows}),
        },
        "split_case_overlap": {},
        "feature_names": {
            name: _feature_names(mode, dataset_vocab, schema_vocab)
            for name, mode in MODEL_FEATURE_MODES.items()
        },
        "parameter_stats_all_rows": _parameter_stats(y_all),
        "parameter_stats_train_rows": _parameter_stats(y_train),
        "models": {},
    }
    split_sets = {split: set(values) for split, values in diagnostics["split_case_ids"].items()}
    for left in ["train", "val", "test"]:
        for right in ["train", "val", "test"]:
            if left < right:
                diagnostics["split_case_overlap"][f"{left}_x_{right}"] = sorted(
                    split_sets[left] & split_sets[right]
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, mode in MODEL_FEATURE_MODES.items():
        val_pred = _fit_predict(
            train_rows,
            val_rows,
            mode,
            min_severity,
            max_severity,
            dataset_vocab,
            schema_vocab,
            args.ridge_alpha,
        )
        test_pred = _fit_predict(
            train_rows,
            test_rows,
            mode,
            min_severity,
            max_severity,
            dataset_vocab,
            schema_vocab,
            args.ridge_alpha,
        )
        test_pred_shuffled_train = _fit_predict(
            train_rows,
            test_rows,
            mode,
            min_severity,
            max_severity,
            dataset_vocab,
            schema_vocab,
            args.ridge_alpha,
            shuffled_train_targets=shuffled_train_targets,
        )
        rng_eval = np.random.default_rng(args.seed)
        shuffled_test_truth = y_test.copy()
        rng_eval.shuffle(shuffled_test_truth, axis=0)

        diagnostics["models"][model_name] = {
            "feature_mode": mode,
            "val": _error_summary(y_val, val_pred, train_std),
            "test": _error_summary(y_test, test_pred, train_std),
            "test_error_against_shuffled_truth": _error_summary(shuffled_test_truth, test_pred, train_std),
            "test_after_shuffled_train_targets": _error_summary(
                y_test, test_pred_shuffled_train, train_std
            ),
            "prediction_plot": str(args.output_dir / f"{model_name}_test_predictions.png"),
        }
        _draw_prediction_plot(
            args.output_dir / f"{model_name}_test_predictions.png",
            y_test,
            test_pred,
        )

    output_path = args.output_dir / "diagnostics.json"
    output_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
