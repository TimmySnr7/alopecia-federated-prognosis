"""Run the Experiment 02 severity-scorer entry gate.

This script evaluates lightweight, reproducible feature-based severity scorers
before any learned residual editor is trained. It is intentionally conservative:
if the exact seven-class scorer cannot clear the macro-F1 gate, downstream
generation should remain blocked or be reported as exploratory only.
"""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _load_rows(manifest_path: Path, view_filter: str | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = Path(row["image_path"])
            if not image_path.exists():
                continue
            if view_filter is not None and row.get("image_view") != view_filter:
                continue
            severity = str(row.get("severity_proxy_value", "")).strip()
            if not severity:
                continue
            try:
                int(float(severity))
            except ValueError:
                continue
            rows.append(row)
    return rows


def _ellipse_mask(size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return (((xx - size / 2) / (size * 0.38)) ** 2 + ((yy - size / 2) / (size * 0.45)) ** 2) <= 1


def _summary_stats(values: np.ndarray) -> list[float]:
    return [
        float(values.mean()),
        float(values.std()),
        *[float(value) for value in np.quantile(values, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])],
    ]


def _extract_features(image_path: str, size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    resized = image.resize((size, size), resample=Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    gray_image = resized.convert("L")
    gray = np.asarray(gray_image, dtype=np.float32) / 255.0
    mask = _ellipse_mask(size)

    features: list[float] = []
    for pixels in [rgb.reshape(-1, 3), rgb[mask].reshape(-1, 3)]:
        features.extend([float(value) for value in pixels.mean(axis=0)])
        features.extend([float(value) for value in pixels.std(axis=0)])
        features.extend([float(value) for value in np.quantile(pixels, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0).ravel()])

    for pixels in [gray.ravel(), gray[mask].ravel()]:
        features.extend(_summary_stats(pixels))
        features.extend([float((pixels < threshold).mean()) for threshold in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]])

    blurred = np.asarray(
        resized.filter(ImageFilter.GaussianBlur(radius=3)).convert("L"),
        dtype=np.float32,
    ) / 255.0
    detail = np.abs(gray - blurred)
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    gradient = np.sqrt(gx * gx + gy * gy)

    for texture in [detail, gradient]:
        for pixels in [texture.ravel(), texture[mask].ravel()]:
            features.extend(_summary_stats(pixels))
            features.extend([float((pixels > threshold).mean()) for threshold in [0.03, 0.06, 0.10, 0.15]])

    width, height = image.size
    features.extend(
        [
            float(width / max(height, 1)),
            float(np.log(width + 1)),
            float(np.log(height + 1)),
        ]
    )
    return np.asarray(features, dtype=np.float32)


def _exact_label(severity: int) -> int:
    return severity


def _ordinal_3_label(severity: int) -> str:
    if severity <= 2:
        return "low_1_2"
    if severity <= 5:
        return "mid_3_5"
    return "high_6_7"


def _build_models(seed: int) -> dict[str, object]:
    return {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=seed),
        "ridge_balanced": make_pipeline(
            StandardScaler(),
            RidgeClassifier(class_weight="balanced"),
        ),
        "logreg_c0_5_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=0.5,
                random_state=seed,
            ),
        ),
        "logreg_c2_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=2.0,
                random_state=seed,
            ),
        ),
        "svc_linear_balanced": make_pipeline(
            StandardScaler(),
            SVC(C=1.0, kernel="linear", class_weight="balanced"),
        ),
        "svc_rbf_balanced": make_pipeline(
            StandardScaler(),
            SVC(C=3.0, kernel="rbf", gamma="scale", class_weight="balanced"),
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=seed,
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=seed,
        ),
    }


def _evaluate_label_mode(
    X_train: np.ndarray,
    X_val: np.ndarray,
    train_severities: list[int],
    val_severities: list[int],
    label_fn: Callable[[int], int | str],
    seed: int,
    macro_f1_gate: float,
) -> dict[str, object]:
    y_train = np.asarray([label_fn(value) for value in train_severities])
    y_val = np.asarray([label_fn(value) for value in val_severities])
    labels = sorted(set(y_train), key=str)

    model_results: list[dict[str, object]] = []
    for model_name, model in _build_models(seed).items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        macro_f1_all = f1_score(
            y_val,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        present_labels = sorted(set(y_val), key=str)
        macro_f1_present = f1_score(
            y_val,
            predictions,
            labels=present_labels,
            average="macro",
            zero_division=0,
        )
        model_results.append(
            {
                "model_name": model_name,
                "accuracy": float(accuracy_score(y_val, predictions)),
                "macro_f1_all_train_labels": float(macro_f1_all),
                "macro_f1_val_present_labels": float(macro_f1_present),
                "predictions": [str(value) for value in predictions.tolist()],
                "confusion_matrix": confusion_matrix(y_val, predictions, labels=labels).tolist(),
                "classification_report": classification_report(
                    y_val,
                    predictions,
                    labels=labels,
                    zero_division=0,
                    output_dict=True,
                ),
            }
        )

    best = max(model_results, key=lambda item: item["macro_f1_all_train_labels"])
    return {
        "labels": [str(label) for label in labels],
        "train_label_distribution": dict(sorted(Counter(str(value) for value in y_train).items())),
        "val_label_distribution": dict(sorted(Counter(str(value) for value in y_val).items())),
        "model_results": model_results,
        "best_model_name": best["model_name"],
        "best_macro_f1_all_train_labels": best["macro_f1_all_train_labels"],
        "best_macro_f1_val_present_labels": best["macro_f1_val_present_labels"],
        "best_accuracy": best["accuracy"],
        "macro_f1_gate": macro_f1_gate,
        "gate_passed": bool(best["macro_f1_all_train_labels"] >= macro_f1_gate),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("train_manifest", type=Path)
    parser.add_argument("val_manifest", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--view-filter", default=None)
    parser.add_argument("--macro-f1-gate", type=float, default=0.40)
    args = parser.parse_args()

    train_rows = _load_rows(args.train_manifest, view_filter=args.view_filter)
    val_rows = _load_rows(args.val_manifest, view_filter=args.view_filter)
    X_train = np.vstack([_extract_features(row["image_path"], args.image_size) for row in train_rows])
    X_val = np.vstack([_extract_features(row["image_path"], args.image_size) for row in val_rows])
    train_severities = [int(float(row["severity_proxy_value"])) for row in train_rows]
    val_severities = [int(float(row["severity_proxy_value"])) for row in val_rows]

    exact = _evaluate_label_mode(
        X_train=X_train,
        X_val=X_val,
        train_severities=train_severities,
        val_severities=val_severities,
        label_fn=_exact_label,
        seed=args.seed,
        macro_f1_gate=args.macro_f1_gate,
    )
    ordinal_3 = _evaluate_label_mode(
        X_train=X_train,
        X_val=X_val,
        train_severities=train_severities,
        val_severities=val_severities,
        label_fn=_ordinal_3_label,
        seed=args.seed,
        macro_f1_gate=args.macro_f1_gate,
    )

    summary = {
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest),
        "view_filter": args.view_filter,
        "image_size": args.image_size,
        "seed": args.seed,
        "train_sample_count": len(train_rows),
        "val_sample_count": len(val_rows),
        "train_severity_distribution": dict(sorted(Counter(train_severities).items())),
        "val_severity_distribution": dict(sorted(Counter(val_severities).items())),
        "feature_count": int(X_train.shape[1]),
        "macro_f1_gate": args.macro_f1_gate,
        "exact_7_class": exact,
        "ordinal_3_class": ordinal_3,
        "experiment_02_scorer_gate_passed": bool(exact["gate_passed"]),
        "gate_interpretation": (
            "Exact seven-class scorer cleared the Experiment 02 entry gate."
            if exact["gate_passed"]
            else "Exact seven-class scorer did not clear the Experiment 02 entry gate; learned residual-editor training should remain blocked or exploratory."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
