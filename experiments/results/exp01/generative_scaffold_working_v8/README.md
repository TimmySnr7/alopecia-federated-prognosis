# Generative Scaffold: Working Subset (v8)

## Purpose

This package records the first Experiment 1 generative run that used an
external frozen severity scorer rather than only the scaffold's internal
severity head.

## Inputs

- Train manifest: `/home/tmushuru/datasets/alopecia_public/manifests/exp01_working_train.csv`
- Validation manifest: `/home/tmushuru/datasets/alopecia_public/manifests/exp01_working_val.csv`
- Config: `experiments/configs/exp01_centralised_baseline.yaml`
- Frozen scorer: `/home/tmushuru/datasets/alopecia_public/metadata/exp01_working_severity_baseline_v4.ckpt`

## Training setup

- Model type: conditional reconstruction scaffold with external severity guidance
- Model name: `ldm-256-cross_attention`
- Image size: `256`
- Condition classes: `1..7`
- Epochs: `15`
- Batch size: `4`
- Severity loss weight: `1.0`
- Train target mode: `sampled`
- Train target shift probability: `0.8`
- Device: `cuda`

## Why this run matters

This run is an explicit diagnostic step. It tests whether a stronger frozen
severity scorer is enough to induce target-sensitive behaviour in the current
public-data scaffold.

## Saved artefacts

- `summary.json`: machine-readable training summary
- `interpretation.md`: concise research interpretation and why `v9` is needed
