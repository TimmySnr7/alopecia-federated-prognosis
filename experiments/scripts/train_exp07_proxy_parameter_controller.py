"""Experiment 07 snapshot: learn proxy-parameter controllers.

This is intentionally lightweight. It tests whether the QA-gated proxy
parameter space can be predicted from severity and metadata before investing in
image-aware controllers or AI5 rendering.
"""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PARAMETER_NAMES = [
    "ellipse_cx",
    "ellipse_cy",
    "ellipse_rx",
    "ellipse_ry",
    "mask_blur_radius",
    "texture_blur_radius",
    "lower_hair_enhancement_eff",
    "lower_tone_shift_eff",
    "upper_hair_suppression_eff",
    "upper_smoothing_eff",
    "upper_tone_shift_eff",
]


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)]
    return [row for row in rows if _parse_bool(row.get("auto_pass", "false"))]


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _load_json_cell(row: dict[str, str], key: str) -> Any:
    return json.loads(row[key])


def _alpha_values(source: float, target: float, min_severity: float, max_severity: float) -> tuple[float, float]:
    if target < source:
        return (source - target) / max(source - min_severity, 1e-8), 0.0
    if target > source:
        return 0.0, (target - source) / max(max_severity - source, 1e-8)
    return 0.0, 0.0


def _target_vector(row: dict[str, str], min_severity: float, max_severity: float) -> np.ndarray:
    source = _float(row, "source_severity")
    target = _float(row, "target_severity")
    alpha_down, alpha_up = _alpha_values(source, target, min_severity, max_severity)
    ellipse = [float(value) for value in _load_json_cell(row, "ellipse_mask")]
    params = _load_json_cell(row, "proxy_parameters")
    return np.asarray(
        [
            *ellipse,
            float(params["mask_blur_radius"]),
            float(params["texture_blur_radius"]),
            alpha_down * float(params["hair_enhancement_strength"]),
            -alpha_down * float(params["tone_shift_strength"]) * 0.5,
            alpha_up * float(params["hair_suppression_strength"]),
            alpha_up * float(params["smoothing_strength"]),
            alpha_up * float(params["tone_shift_strength"]),
        ],
        dtype=np.float64,
    )


def _fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def _standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean) / std


def _one_hot(value: str, vocabulary: list[str]) -> list[float]:
    return [1.0 if value == item else 0.0 for item in vocabulary]


def _numeric_features(row: dict[str, str], min_severity: float, max_severity: float) -> list[float]:
    source = _float(row, "source_severity")
    target = _float(row, "target_severity")
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


def _build_features(
    rows: list[dict[str, str]],
    *,
    mode: str,
    min_severity: float,
    max_severity: float,
    dataset_vocab: list[str],
    schema_vocab: list[str],
) -> np.ndarray:
    features: list[list[float]] = []
    for row in rows:
        if mode == "constant":
            vector = [1.0]
        elif mode == "severity":
            vector = _numeric_features(row, min_severity, max_severity)
        elif mode == "metadata":
            vector = [
                *_numeric_features(row, min_severity, max_severity),
                *_one_hot(row.get("dataset_key", ""), dataset_vocab),
                *_one_hot(row.get("label_schema", ""), schema_vocab),
            ]
        else:
            raise ValueError(f"Unsupported feature mode: {mode}")
        features.append(vector)
    return np.asarray(features, dtype=np.float64)


def _fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(X.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.pinv(X.T @ X + penalty) @ X.T @ Y


def _predict_constant(train_targets: np.ndarray, count: int) -> np.ndarray:
    return np.repeat(train_targets.mean(axis=0, keepdims=True), count, axis=0)


def _parameter_errors(y_true: np.ndarray, y_pred: np.ndarray, train_std: np.ndarray) -> dict[str, Any]:
    abs_error = np.abs(y_pred - y_true)
    normalized = abs_error / train_std
    return {
        "mae": float(abs_error.mean()),
        "normalized_mae": float(normalized.mean()),
        "per_parameter_mae": {
            name: float(value) for name, value in zip(PARAMETER_NAMES, abs_error.mean(axis=0))
        },
        "per_parameter_normalized_mae": {
            name: float(value) for name, value in zip(PARAMETER_NAMES, normalized.mean(axis=0))
        },
    }


def _effect_index(vector: np.ndarray) -> float:
    by_name = dict(zip(PARAMETER_NAMES, vector))
    return float(
        by_name["upper_hair_suppression_eff"]
        + by_name["upper_smoothing_eff"]
        + by_name["upper_tone_shift_eff"]
        - by_name["lower_hair_enhancement_eff"]
        + by_name["lower_tone_shift_eff"]
    )


def _monotonic_fraction(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    return sum(next_value >= value for value, next_value in zip(values, values[1:])) / (len(values) - 1)


def _mask_area_from_ellipse(vector: np.ndarray) -> float:
    _, _, rx, ry = vector[:4]
    return float(np.pi * max(rx, 0.0) * max(ry, 0.0))


def _case_diagnostics(rows: list[dict[str, str]], predictions: np.ndarray) -> dict[str, Any]:
    grouped: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        grouped[row["case_id"]].append((int(float(row["target_severity"])), prediction))

    case_results: list[dict[str, Any]] = []
    for case_id, items in sorted(grouped.items()):
        ordered = [prediction for _, prediction in sorted(items, key=lambda item: item[0])]
        effects = [_effect_index(prediction) for prediction in ordered]
        mask_areas = [_mask_area_from_ellipse(prediction) for prediction in ordered]
        case_results.append(
            {
                "case_id": case_id,
                "predicted_effect_monotonic_fraction": float(_monotonic_fraction(effects)),
                "predicted_effect_span": float(max(effects) - min(effects)) if effects else 0.0,
                "predicted_mask_area_min": float(min(mask_areas)) if mask_areas else 0.0,
                "predicted_mask_area_max": float(max(mask_areas)) if mask_areas else 0.0,
                "predicted_mask_area_in_gate": bool(
                    all(0.20 <= value <= 0.75 for value in mask_areas)
                ),
            }
        )

    return {
        "case_count": len(case_results),
        "mean_predicted_effect_monotonic_fraction": float(
            np.mean([item["predicted_effect_monotonic_fraction"] for item in case_results])
        )
        if case_results
        else 0.0,
        "mean_predicted_effect_span": float(
            np.mean([item["predicted_effect_span"] for item in case_results])
        )
        if case_results
        else 0.0,
        "mask_area_gate_pass_rate": float(
            np.mean([item["predicted_mask_area_in_gate"] for item in case_results])
        )
        if case_results
        else 0.0,
        "cases": case_results,
    }


def _write_predictions(path: Path, rows: list[dict[str, str]], y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    fieldnames = [
        "pair_id",
        "case_id",
        "split",
        "dataset_key",
        "source_severity",
        "target_severity",
    ]
    for prefix in ["true", *predictions.keys()]:
        fieldnames.extend([f"{prefix}_{name}" for name in PARAMETER_NAMES])

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            output = {
                "pair_id": row["pair_id"],
                "case_id": row["case_id"],
                "split": row["split"],
                "dataset_key": row["dataset_key"],
                "source_severity": row["source_severity"],
                "target_severity": row["target_severity"],
            }
            for name, value in zip(PARAMETER_NAMES, y_true[index]):
                output[f"true_{name}"] = f"{value:.8f}"
            for model_name, values in predictions.items():
                for name, value in zip(PARAMETER_NAMES, values[index]):
                    output[f"{model_name}_{name}"] = f"{value:.8f}"
            writer.writerow(output)


def _summarize_split(name: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "split": name,
        "pair_count": len(rows),
        "case_count": len({row["case_id"] for row in rows}),
        "dataset_counts": dict(sorted({key: sum(row["dataset_key"] == key for row in rows) for key in {row["dataset_key"] for row in rows}}.items())),
    }


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
        default=Path("experiments/results/exp07/proxy_parameter_controller_snapshot_v1"),
    )
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    args = parser.parse_args()

    rows = _load_rows(args.pair_manifest)
    if not rows:
        raise ValueError(f"No auto-pass rows found in {args.pair_manifest}")

    severities = [float(row["source_severity"]) for row in rows] + [
        float(row["target_severity"]) for row in rows
    ]
    min_severity = min(severities)
    max_severity = max(severities)

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if not train_rows or not val_rows or not test_rows:
        raise ValueError("Expected train, val, and test rows in the pair manifest.")

    dataset_vocab = sorted({row.get("dataset_key", "") for row in train_rows})
    schema_vocab = sorted({row.get("label_schema", "") for row in train_rows})

    y_train = np.vstack([_target_vector(row, min_severity, max_severity) for row in train_rows])
    y_val = np.vstack([_target_vector(row, min_severity, max_severity) for row in val_rows])
    y_test = np.vstack([_target_vector(row, min_severity, max_severity) for row in test_rows])
    y_mean, y_std = _fit_standardizer(y_train)

    results: dict[str, Any] = {
        "experiment": "exp07_proxy_parameter_controller_snapshot_v1",
        "pair_manifest": str(args.pair_manifest),
        "parameter_names": PARAMETER_NAMES,
        "target_definition": "effective proxy vector: ellipse, blur constants, and severity-scaled edit strengths",
        "image_features_available_locally": False,
        "image_feature_note": "Manifest image paths point to AI5 /home/tmushuru paths and were not present locally; this snapshot evaluates severity/metadata controllers only.",
        "splits": [
            _summarize_split("train", train_rows),
            _summarize_split("val", val_rows),
            _summarize_split("test", test_rows),
        ],
        "models": {},
    }

    all_rows = [*train_rows, *val_rows, *test_rows]
    all_true = np.vstack([y_train, y_val, y_test])
    all_predictions: dict[str, np.ndarray] = {}

    for model_name, feature_mode in [
        ("constant_mean", "constant"),
        ("severity_ridge", "severity"),
        ("metadata_ridge", "metadata"),
    ]:
        if model_name == "constant_mean":
            val_pred = _predict_constant(y_train, len(val_rows))
            test_pred = _predict_constant(y_train, len(test_rows))
            all_pred = _predict_constant(y_train, len(all_rows))
        else:
            X_train = _build_features(
                train_rows,
                mode=feature_mode,
                min_severity=min_severity,
                max_severity=max_severity,
                dataset_vocab=dataset_vocab,
                schema_vocab=schema_vocab,
            )
            x_mean, x_std = _fit_standardizer(X_train[:, 1:] if X_train.shape[1] > 1 else X_train)
            if X_train.shape[1] > 1:
                X_train_scaled = np.column_stack([X_train[:, 0], _standardize(X_train[:, 1:], x_mean, x_std)])
            else:
                X_train_scaled = X_train
            weights = _fit_ridge(X_train_scaled, _standardize(y_train, y_mean, y_std), args.ridge_alpha)

            def predict(split_rows: list[dict[str, str]]) -> np.ndarray:
                X = _build_features(
                    split_rows,
                    mode=feature_mode,
                    min_severity=min_severity,
                    max_severity=max_severity,
                    dataset_vocab=dataset_vocab,
                    schema_vocab=schema_vocab,
                )
                if X.shape[1] > 1:
                    X = np.column_stack([X[:, 0], _standardize(X[:, 1:], x_mean, x_std)])
                return (X @ weights) * y_std + y_mean

            val_pred = predict(val_rows)
            test_pred = predict(test_rows)
            all_pred = predict(all_rows)

        results["models"][model_name] = {
            "feature_mode": feature_mode,
            "val": {
                **_parameter_errors(y_val, val_pred, y_std),
                "controller_diagnostics": _case_diagnostics(val_rows, val_pred),
            },
            "test": {
                **_parameter_errors(y_test, test_pred, y_std),
                "controller_diagnostics": _case_diagnostics(test_rows, test_pred),
            },
        }
        all_predictions[model_name] = all_pred

    best_model = min(
        results["models"].items(),
        key=lambda item: item[1]["test"]["normalized_mae"],
    )
    results["best_by_test_normalized_mae"] = {
        "model_name": best_model[0],
        "test_normalized_mae": best_model[1]["test"]["normalized_mae"],
        "test_mae": best_model[1]["test"]["mae"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    predictions_path = args.output_dir / "parameter_predictions.csv"
    summary_path.write_text(json.dumps(results, indent=2))
    _write_predictions(predictions_path, all_rows, all_true, all_predictions)
    print(json.dumps(results["best_by_test_normalized_mae"], indent=2))
    print(f"Wrote {summary_path}")
    print(f"Wrote {predictions_path}")


if __name__ == "__main__":
    main()
