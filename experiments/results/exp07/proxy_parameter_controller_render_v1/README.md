# Experiment 07 Render v1 — Learned Proxy-Parameter Controller

This run evaluates Experiment 07 on AI5 with image rendering and the full computational QA gate.

## Purpose

Experiments 03--06 showed that direct learned image generation is not reliable at the current data scale. Experiment 07 tests a lower-risk formulation: learn proxy-control parameters and let the deterministic Experiment 02 proxy renderer synthesize the final image.

## Command

```bash
cd ~/github-repos/alopecia-federated-prognosis
source .venv/bin/activate
python experiments/scripts/run_exp07_proxy_parameter_controller_render.py \
  --scorer-checkpoint /home/tmushuru/datasets/alopecia_public/metadata/exp05_proxy_severity_scorer_v1.ckpt \
  --output-dir experiments/results/exp07/proxy_parameter_controller_render_v1
```

## Models

- `constant_mean`: predicts the train-set mean parameter vector.
- `severity_ridge`: predicts effective proxy parameters from source severity, target severity, severity delta, and direction scalars.
- `metadata_ridge`: adds dataset/schema metadata to the severity features.

## Headline Results

| Model | Val auto-pass | Test auto-pass | Val span | Test span | Test unmasked drift |
|---|---:|---:|---:|---:|---:|
| Constant mean | 0.000 | 0.000 | 0.000 | 0.000 | 0.0020 |
| Severity ridge | 1.000 | 1.000 | 5.651 | 5.346 | 0.0055 |
| Metadata ridge | 1.000 | 1.000 | 4.458 | 5.349 | 0.0061 |

## Interpretation

The Experiment 07 pivot is promising. Both learned controllers pass the full auto-pass gate on validation and test after rendering. This is the first learned formulation after Experiment 02 to recover proxy-level monotonicity and span without hallucination, because it learns control parameters rather than pixels.

The stronger result is `metadata_ridge`, because it recovers the dataset/mask-preset differences and nearly reproduces the Experiment 02 proxy reference. The `severity_ridge` model also passes all gates, but on validation it predicts larger masks than the proxy reference because it lacks dataset-specific mask information.

## Contact Sheets

- `val_metadata_ridge_contact_sheet.png`
- `test_metadata_ridge_contact_sheet.png`
- `val_severity_ridge_contact_sheet.png`
- `test_severity_ridge_contact_sheet.png`
- `val_constant_mean_contact_sheet.png`
- `test_constant_mean_contact_sheet.png`

## Limitation

This is a positive computational result, not a clinical validation. The current controller mostly learns the existing deterministic proxy parameterisation. The next scientific step is to make the proxy parameter space richer or image-aware, then test whether source-image features improve parameter choice beyond severity and metadata.
