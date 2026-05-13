"""Batch-evaluate the Experiment 1 plausible proxy sweep with mask QA panels."""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any

from PIL import Image, ImageDraw, ImageFilter
import torch
from torchvision.transforms import InterpolationMode, Resize, ToTensor

from experiments.scripts.run_exp01_plausible_proxy_sweep import (
    _add_label,
    _build_ellipse_mask,
    _expected_severity,
    _make_proxy_edit,
    _monotonic_fraction,
    _to_pil,
    _visual_delta_metrics,
    _load_scorer,
)


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return safe.strip("_") or "case"


def _load_manifest_rows(
    manifest_path: Path,
    view: str,
    max_samples: int | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if view and row.get("image_view") != view:
                continue
            image_path = Path(row["image_path"])
            if not image_path.exists():
                continue
            severity = str(row.get("severity_proxy_value", "")).strip()
            if not severity:
                continue
            try:
                int(float(severity))
            except ValueError:
                continue
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def _score_image(
    scorer: torch.nn.Module,
    image: torch.Tensor,
    labels: list[int],
) -> dict[str, float | int]:
    logits = scorer(image)
    probabilities = torch.softmax(logits[0], dim=0)
    predicted_index = int(probabilities.argmax().item())
    predicted_label = int(labels[predicted_index])
    confidence = float(probabilities[predicted_index].item())
    expected = _expected_severity(probabilities, labels)
    return {
        "predicted_severity": predicted_label,
        "predicted_confidence": confidence,
        "expected_severity": expected,
    }


def _compose_horizontal(panels: list[Image.Image]) -> Image.Image:
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * width, 0))
    return canvas


def _compose_grid(panels: list[Image.Image], columns: int) -> Image.Image:
    if not panels:
        raise ValueError("Cannot compose an empty grid.")
    width, height = panels[0].size
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (width * columns, height * rows), (255, 255, 255))
    for index, panel in enumerate(panels):
        x = (index % columns) * width
        y = (index // columns) * height
        canvas.paste(panel, (x, y))
    return canvas


def _mask_overlay(source: torch.Tensor, mask: torch.Tensor) -> Image.Image:
    base = _to_pil(source[0])
    overlay = Image.new("RGB", base.size, (255, 64, 32))
    mask_image = _to_pil(mask[0].repeat(3, 1, 1)).convert("L")
    return Image.blend(base, Image.composite(overlay, base, mask_image), alpha=0.35)


def _load_mask_presets(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text())


def _select_ellipse_mask(
    row: dict[str, str],
    default_ellipse: list[float],
    mask_presets: dict[str, Any],
) -> tuple[list[float], str]:
    case_presets = mask_presets.get("by_case_id", {})
    image_path = Path(row["image_path"])
    for key in [str(image_path), image_path.name, image_path.stem]:
        if key in case_presets:
            return [float(value) for value in case_presets[key]], f"case:{key}"

    dataset_presets = mask_presets.get("by_dataset_key", {})
    dataset_key = row.get("dataset_key", "")
    if dataset_key in dataset_presets:
        return [float(value) for value in dataset_presets[dataset_key]], f"dataset:{dataset_key}"

    return default_ellipse, "default"


def _case_passes(
    case_summary: dict[str, Any],
    min_monotonic_fraction: float,
    min_expected_span: float,
    max_unmasked_delta: float,
    mask_area_min: float,
    mask_area_max: float,
) -> bool:
    return (
        case_summary["expected_severity_monotonic_fraction"] >= min_monotonic_fraction
        and case_summary["expected_severity_span"] >= min_expected_span
        and case_summary["max_unmasked_mean_absolute_delta"] <= max_unmasked_delta
        and mask_area_min <= case_summary["mask_area_fraction"] <= mask_area_max
    )


def _flatten_case(case_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case_summary["case_id"],
        "dataset_key": case_summary["dataset_key"],
        "split": case_summary["split"],
        "image_view": case_summary["image_view"],
        "source_severity": case_summary["source_severity"],
        "source_predicted_severity": case_summary["source_score"]["predicted_severity"],
        "source_expected_severity": case_summary["source_score"]["expected_severity"],
        "expected_severity_monotonic_fraction": case_summary[
            "expected_severity_monotonic_fraction"
        ],
        "predicted_severity_monotonic_fraction": case_summary[
            "predicted_severity_monotonic_fraction"
        ],
        "expected_severity_span": case_summary["expected_severity_span"],
        "mask_area_fraction": case_summary["mask_area_fraction"],
        "max_mean_absolute_delta": case_summary["max_mean_absolute_delta"],
        "max_masked_mean_absolute_delta": case_summary["max_masked_mean_absolute_delta"],
        "max_unmasked_mean_absolute_delta": case_summary["max_unmasked_mean_absolute_delta"],
        "auto_pass": case_summary["auto_pass"],
        "panel_path": case_summary["panel_path"],
        "image_path": case_summary["image_path"],
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("scorer_checkpoint", type=Path)
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view", default="top")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--ellipse-mask",
        nargs=4,
        type=float,
        metavar=("CX", "CY", "RX", "RY"),
        default=[0.50, 0.52, 0.36, 0.45],
        help="Normalized ellipse mask in resized-image coordinates.",
    )
    parser.add_argument("--mask-blur-radius", type=float, default=10.0)
    parser.add_argument("--mask-preset-json", type=Path, default=None)
    parser.add_argument("--texture-blur-radius", type=float, default=3.0)
    parser.add_argument("--hair-enhancement-strength", type=float, default=1.2)
    parser.add_argument("--hair-suppression-strength", type=float, default=1.5)
    parser.add_argument("--smoothing-strength", type=float, default=0.35)
    parser.add_argument("--tone-shift-strength", type=float, default=0.03)
    parser.add_argument("--min-monotonic-fraction", type=float, default=0.8)
    parser.add_argument("--min-expected-span", type=float, default=0.5)
    parser.add_argument("--max-unmasked-delta", type=float, default=0.01)
    parser.add_argument("--mask-area-min", type=float, default=0.2)
    parser.add_argument("--mask-area-max", type=float, default=0.75)
    parser.add_argument("--contact-sheet-columns", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = args.output_dir / "case_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer, labels, image_size = _load_scorer(args.scorer_checkpoint, device)
    resize = Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
    rows = _load_manifest_rows(
        manifest_path=args.manifest_path,
        view=args.view,
        max_samples=args.max_samples,
    )
    mask_presets = _load_mask_presets(args.mask_preset_json)

    case_summaries: list[dict[str, Any]] = []
    contact_panels: list[Image.Image] = []
    mask_qa_panels: list[Image.Image] = []
    min_severity = min(labels)
    max_severity = max(labels)

    with torch.no_grad():
        for row_index, row in enumerate(rows):
            source_severity = int(float(row["severity_proxy_value"]))
            image_path = Path(row["image_path"])
            case_id = f"{row_index:03d}_{_safe_id(row['dataset_key'])}_{source_severity}_{_safe_id(image_path.stem)}"

            image = Image.open(image_path).convert("RGB")
            resized_image = resize(image)
            blurred_image = resized_image.filter(
                ImageFilter.GaussianBlur(radius=args.texture_blur_radius)
            )
            source = ToTensor()(resized_image).unsqueeze(0).to(device)
            blurred = ToTensor()(blurred_image).unsqueeze(0).to(device)
            selected_ellipse, selected_ellipse_source = _select_ellipse_mask(
                row=row,
                default_ellipse=args.ellipse_mask,
                mask_presets=mask_presets,
            )
            mask = _build_ellipse_mask(
                selected_ellipse,
                image_size=image_size,
                device=device,
                blur_radius=args.mask_blur_radius,
            )
            source_score = _score_image(scorer, source, labels)

            sweep_panels = [
                _add_label(
                    _to_pil(source[0]),
                    [
                        f"input sev {source_severity}",
                        f"pred {source_score['predicted_severity']}",
                        f"exp {source_score['expected_severity']:.2f}",
                    ],
                )
            ]
            target_summaries: list[dict[str, Any]] = []
            for target_severity in labels:
                edited = _make_proxy_edit(
                    source=source,
                    blurred=blurred,
                    mask=mask,
                    target_severity=target_severity,
                    source_severity=float(source_severity),
                    min_severity=min_severity,
                    max_severity=max_severity,
                    hair_enhancement_strength=args.hair_enhancement_strength,
                    hair_suppression_strength=args.hair_suppression_strength,
                    smoothing_strength=args.smoothing_strength,
                    tone_shift_strength=args.tone_shift_strength,
                )
                score = _score_image(scorer, edited, labels)
                target_summary = {
                    "target_severity": int(target_severity),
                    **score,
                    **_visual_delta_metrics(source, edited, mask),
                }
                target_summaries.append(target_summary)
                sweep_panels.append(
                    _add_label(
                        _to_pil(edited[0]),
                        [
                            f"target {target_severity}",
                            f"pred {score['predicted_severity']} ({score['predicted_confidence']:.2f})",
                            f"exp {score['expected_severity']:.2f}",
                        ],
                    )
                )

            expected_values = [item["expected_severity"] for item in target_summaries]
            predicted_values = [item["predicted_severity"] for item in target_summaries]
            case_panel = _compose_horizontal(sweep_panels)
            panel_path = panels_dir / f"{case_id}.png"
            case_panel.save(panel_path)

            case_summary: dict[str, Any] = {
                "case_id": case_id,
                "dataset_key": row["dataset_key"],
                "split": row.get("split", ""),
                "image_view": row.get("image_view", ""),
                "label_schema": row.get("label_schema", ""),
                "image_path": str(image_path),
                "source_severity": source_severity,
                "source_score": source_score,
                "ellipse_mask": selected_ellipse,
                "ellipse_mask_source": selected_ellipse_source,
                "mask_area_fraction": target_summaries[0]["mask_area_fraction"],
                "targets": target_summaries,
                "expected_severity_monotonic_fraction": _monotonic_fraction(expected_values),
                "predicted_severity_monotonic_fraction": _monotonic_fraction(predicted_values),
                "expected_severity_span": max(expected_values) - min(expected_values),
                "max_mean_absolute_delta": max(
                    item["mean_absolute_delta"] for item in target_summaries
                ),
                "max_masked_mean_absolute_delta": max(
                    item["masked_mean_absolute_delta"] for item in target_summaries
                ),
                "max_unmasked_mean_absolute_delta": max(
                    item["unmasked_mean_absolute_delta"] for item in target_summaries
                ),
                "panel_path": str(panel_path),
            }
            case_summary["auto_pass"] = _case_passes(
                case_summary,
                min_monotonic_fraction=args.min_monotonic_fraction,
                min_expected_span=args.min_expected_span,
                max_unmasked_delta=args.max_unmasked_delta,
                mask_area_min=args.mask_area_min,
                mask_area_max=args.mask_area_max,
            )
            case_summaries.append(case_summary)
            contact_panels.append(case_panel)
            mask_qa_panels.extend(
                [
                    _add_label(
                        _to_pil(source[0]),
                        [
                            case_id[:20],
                            f"sev {source_severity}",
                            row["dataset_key"][:18],
                        ],
                    ),
                    _add_label(
                        _mask_overlay(source, mask),
                        [
                            "mask overlay",
                            f"area {case_summary['mask_area_fraction']:.2f}",
                            f"pass {case_summary['auto_pass']}",
                        ],
                    ),
                ]
            )

    flat_cases = [_flatten_case(case_summary) for case_summary in case_summaries]
    csv_path = args.output_dir / "cases.csv"
    if flat_cases:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_cases[0].keys()))
            writer.writeheader()
            writer.writerows(flat_cases)

    contact_sheet_path = args.output_dir / "proxy_sweep_contact_sheet.png"
    mask_qa_path = args.output_dir / "mask_qa_contact_sheet.png"
    if contact_panels:
        _compose_grid(contact_panels, columns=args.contact_sheet_columns).save(contact_sheet_path)
    if mask_qa_panels:
        _compose_grid(mask_qa_panels, columns=4).save(mask_qa_path)

    pass_count = sum(1 for case_summary in case_summaries if case_summary["auto_pass"])
    summary = {
        "scorer_checkpoint": str(args.scorer_checkpoint),
        "manifest_path": str(args.manifest_path),
        "view": args.view,
        "device": str(device),
        "image_size": image_size,
        "condition_classes": labels,
        "case_count": len(case_summaries),
        "auto_pass_count": pass_count,
        "auto_pass_rate": pass_count / len(case_summaries) if case_summaries else 0.0,
        "mean_expected_severity_monotonic_fraction": mean(
            case["expected_severity_monotonic_fraction"] for case in case_summaries
        )
        if case_summaries
        else 0.0,
        "mean_expected_severity_span": mean(
            case["expected_severity_span"] for case in case_summaries
        )
        if case_summaries
        else 0.0,
        "criteria": {
            "min_monotonic_fraction": args.min_monotonic_fraction,
            "min_expected_span": args.min_expected_span,
            "max_unmasked_delta": args.max_unmasked_delta,
            "mask_area_min": args.mask_area_min,
            "mask_area_max": args.mask_area_max,
        },
        "proxy_parameters": {
            "ellipse_mask": args.ellipse_mask,
            "mask_preset_json": str(args.mask_preset_json) if args.mask_preset_json else None,
            "mask_blur_radius": args.mask_blur_radius,
            "texture_blur_radius": args.texture_blur_radius,
            "hair_enhancement_strength": args.hair_enhancement_strength,
            "hair_suppression_strength": args.hair_suppression_strength,
            "smoothing_strength": args.smoothing_strength,
            "tone_shift_strength": args.tone_shift_strength,
        },
        "cases_csv": str(csv_path),
        "contact_sheet": str(contact_sheet_path),
        "mask_qa_contact_sheet": str(mask_qa_path),
        "cases": case_summaries,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
