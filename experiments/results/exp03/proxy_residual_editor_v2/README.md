# Proxy Residual Editor (v2)

## Purpose

This run tested whether the Experiment 03 residual editor could recover
target-severity separation after the v1 editor collapsed to muted,
source-preserving edits.

The v2 intervention added severity-delta weighting, dataset-balanced sampling,
stronger masked residual reconstruction, a larger residual scale, and weak
scorer-space consistency against the proxy target.

## Inputs

- Proxy-pair manifest:
  `experiments/results/exp02/source_aware_proxy_batch_v1/proxy_pair_manifest.csv`
- Scorer checkpoint:
  `/home/tmushuru/datasets/alopecia_public/metadata/exp01_working_severity_baseline_v4.ckpt`
- Training pairs: `42`
- Validation pairs: `12`
- Test pairs: `24`
- Training cases: `7`
- Validation cases: `2`
- Test cases: `4`

The manifest contains only Experiment 02 QA-passing cases. The two Experiment
02 proxy-QA failures were excluded before Experiment 03, so the proxy baseline
is evaluated only on its QA-gated subset. The split is source-case-level and
case-disjoint; no source image appears in more than one split.

## Model And Training

- Model: `masked_residual_editor`
- Hidden channels: `32`
- Residual scale: `0.5`
- Texture channels: enabled (`6`)
- Epochs: `75`
- Batch size: `8`
- Sampling: dataset-balanced weighted sampling
- Loss terms: global proxy L1, masked proxy L1, masked residual-delta L1,
  unmasked source-drift penalty, proxy scorer-KL consistency, and proxy
  expected-severity consistency

## Results

| Split | Learned pass rate | Proxy pass rate | Learned mean monotonicity | Proxy mean monotonicity | Learned mean span | Proxy mean span | Mean learned L1 to proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | `0.000` | `1.000` | `0.917` | `0.917` | `0.300` | `1.816` | `0.00548` |
| Test | `0.000` | `1.000` | `0.667` | `1.000` | `0.043` | `2.276` | `0.01330` |

Final pair-level test metrics:

- Final test proxy L1: `0.01459`
- Final test masked proxy L1: `0.02466`
- Final test unmasked source delta: `0.00228`
- Final test scorer KL: `0.32429`
- Final test expected-severity MSE: `1.35064`

## Contact Sheets

The generated visual QA artefacts are intentionally not tracked in Git:

- `/home/tmushuru/datasets/alopecia_public/metadata/exp03_proxy_residual_editor_v2_samples/val_learned_vs_proxy_contact_sheet.png`
- `/home/tmushuru/datasets/alopecia_public/metadata/exp03_proxy_residual_editor_v2_samples/test_learned_vs_proxy_contact_sheet.png`

Local copies may exist under `exp03_proxy_residual_editor_v2_samples/` for
inspection.

## Saved Artefacts

- `summary.json`: full machine-readable training, validation, test, and sweep
  metrics.
- `interpretation.md`: concise research interpretation and final Experiment 03
  decision.
