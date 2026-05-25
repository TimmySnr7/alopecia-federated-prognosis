# Experiment 07 v2 — Image-Aware Proxy-Parameter Controller

Experiment 07 v2 tests whether image-aware controls add value once the proxy parameter space is deliberately enriched with source-image statistics.

## Method

The target vector extends the Experiment 07 v1 effective proxy-control vector with image-dependent edit-strength multipliers derived from masked scalp statistics:

- masked grayscale mean and standard deviation,
- dark-pixel fraction,
- bright-pixel fraction,
- texture mean and 90th percentile,
- unmasked grayscale mean,
- mask area,
- a derived hair-presence index.

The renderer remains deterministic. The learned model predicts parameters; the renderer produces the image; the existing QA gate evaluates the result.

## Models

- `constant_mean`
- `severity_ridge`
- `metadata_ridge`
- `image_stats_ridge`

## Headline Results

| Model | Val parameter MAE | Test parameter MAE | Val auto-pass | Test auto-pass | Test monotonicity | Test span | Test drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Constant mean | 0.1273 | 0.1032 | 0.000 | 0.000 | 1.000 | 0.000 | 0.0020 |
| Severity ridge | 0.0535 | 0.0167 | 1.000 | 1.000 | 1.000 | 5.350 | 0.0053 |
| Metadata ridge | 0.0190 | 0.0123 | 1.000 | 1.000 | 1.000 | 5.352 | 0.0059 |
| Image-stats ridge | 0.0110 | 0.0084 | 1.000 | 1.000 | 0.958 | 5.359 | 0.0053 |

## Interpretation

The richer target confirms that image-aware features can reduce absolute parameter error relative to metadata-only control. However, the improvement is not decisive at image-level QA: all learned controllers pass the auto-pass gate on validation and test, and metadata-ridge preserves perfect monotonicity while image-stats ridge has one mild test inversion.

This means image-aware control is technically feasible but not yet necessary for the current proxy renderer. The proxy parameter space remains too rule-structured for image features to deliver a strong practical advantage.

## Outputs

- `summary.json`
- `val_*_contact_sheet.png`
- `test_*_contact_sheet.png`
