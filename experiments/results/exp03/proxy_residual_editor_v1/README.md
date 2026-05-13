# Proxy Residual Editor (v1)

## Purpose

This run is the first learned residual-editor attempt using the Experiment 02
QA-gated proxy-pair manifest.

The goal was to test whether a small mask-constrained neural editor could learn
the morphology-inspired proxy edits while preserving background/source identity.

## Inputs

- Proxy-pair manifest: `experiments/results/exp02/source_aware_proxy_batch_v1/proxy_pair_manifest.csv`
- Scorer checkpoint: `/home/tmushuru/datasets/alopecia_public/metadata/exp01_working_severity_baseline_v4.ckpt`
- Training pairs: `42`
- Validation pairs: `12`
- Test pairs: `24`
- Training cases: `7`
- Validation cases: `2`
- Test cases: `4`

## Model And Training

- Model: `masked_residual_editor`
- Hidden channels: `32`
- Residual scale: `0.35`
- Texture channels: enabled (`6`)
- Epochs: `50`
- Batch size: `8`
- Augmentation: random horizontal flip
- Loss terms: global proxy L1, masked proxy L1, masked residual-delta L1,
  and high-weight unmasked source-drift penalty

## Results

| Split | Learned pass rate | Proxy pass rate | Learned mean monotonicity | Proxy mean monotonicity | Learned mean span | Proxy mean span | Mean learned L1 to proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | `0.000` | `1.000` | `1.000` | `0.917` | `0.291` | `1.816` | `0.00284` |
| Test | `0.250` | `1.000` | `0.833` | `1.000` | `0.301` | `2.276` | `0.00661` |

Final pair-level metrics:

- Final validation proxy L1: `0.00320`
- Final validation unmasked source delta: `0.000132`
- Final test proxy L1: `0.00717`
- Final test unmasked source delta: `0.00101`

## Contact Sheets

The generated visual QA artefacts are intentionally not tracked in Git:

- `/home/tmushuru/datasets/alopecia_public/metadata/exp03_proxy_residual_editor_v1_samples/val_learned_vs_proxy_contact_sheet.png`
- `/home/tmushuru/datasets/alopecia_public/metadata/exp03_proxy_residual_editor_v1_samples/test_learned_vs_proxy_contact_sheet.png`

Local copies may exist under `exp03_proxy_residual_editor_v1_samples/` for
inspection.

## Saved Artefacts

- `summary.json`: full machine-readable training, validation, test, and sweep
  metrics.
- `interpretation.md`: concise research interpretation and next-step decision.
