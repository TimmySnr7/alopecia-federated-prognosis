"""Export QA-gated Experiment 2 proxy-pair specifications from batch QA output."""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
from typing import Any


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _iter_pair_rows(
    summary: dict[str, Any],
    *,
    include_failures: bool,
    include_identity: bool,
) -> list[dict[str, str]]:
    proxy_parameters = summary.get("proxy_parameters", {})
    rows: list[dict[str, str]] = []
    for case in summary["cases"]:
        auto_pass = bool(case["auto_pass"])
        if not auto_pass and not include_failures:
            continue

        source_severity = int(case["source_severity"])
        for target in case["targets"]:
            target_severity = int(target["target_severity"])
            if target_severity == source_severity and not include_identity:
                continue

            pair_type = "identity" if target_severity == source_severity else "severity_shift"
            rows.append(
                {
                    "pair_id": f"{case['case_id']}__target_{target_severity}",
                    "case_id": case["case_id"],
                    "pair_type": pair_type,
                    "dataset_key": case["dataset_key"],
                    "split": case["split"],
                    "image_view": case["image_view"],
                    "label_schema": case.get("label_schema", ""),
                    "image_path": case["image_path"],
                    "source_severity": str(source_severity),
                    "target_severity": str(target_severity),
                    "source_predicted_severity": str(case["source_score"]["predicted_severity"]),
                    "source_expected_severity": str(case["source_score"]["expected_severity"]),
                    "target_predicted_severity": str(target["predicted_severity"]),
                    "target_expected_severity": str(target["expected_severity"]),
                    "target_predicted_confidence": str(target["predicted_confidence"]),
                    "target_mean_absolute_delta": str(target["mean_absolute_delta"]),
                    "target_masked_mean_absolute_delta": str(target["masked_mean_absolute_delta"]),
                    "target_unmasked_mean_absolute_delta": str(target["unmasked_mean_absolute_delta"]),
                    "case_expected_severity_monotonic_fraction": str(
                        case["expected_severity_monotonic_fraction"]
                    ),
                    "case_predicted_severity_monotonic_fraction": str(
                        case["predicted_severity_monotonic_fraction"]
                    ),
                    "case_expected_severity_span": str(case["expected_severity_span"]),
                    "case_mask_area_fraction": str(case["mask_area_fraction"]),
                    "case_max_unmasked_mean_absolute_delta": str(
                        case["max_unmasked_mean_absolute_delta"]
                    ),
                    "ellipse_mask": _json_cell(case["ellipse_mask"]),
                    "ellipse_mask_source": case.get("ellipse_mask_source", "unknown"),
                    "proxy_parameters": _json_cell(proxy_parameters),
                    "auto_pass": str(auto_pass),
                }
            )
    return rows


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help="Include cases that failed the automatic batch QA criteria.",
    )
    parser.add_argument(
        "--include-identity",
        action="store_true",
        help="Include no-op source-to-source target rows.",
    )
    args = parser.parse_args()

    summary = json.loads(args.summary_json.read_text())
    rows = _iter_pair_rows(
        summary,
        include_failures=args.include_failures,
        include_identity=args.include_identity,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No proxy-pair rows were exported.")

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    exported_case_ids = {row["case_id"] for row in rows}
    print(
        json.dumps(
            {
                "summary_json": str(args.summary_json),
                "output_csv": str(args.output_csv),
                "exported_pair_count": len(rows),
                "exported_case_count": len(exported_case_ids),
                "include_failures": args.include_failures,
                "include_identity": args.include_identity,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
