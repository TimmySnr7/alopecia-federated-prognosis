# Generative Scaffold: Working Subset (v1)

## Purpose

This package records the first successful end-to-end centralised generative
training scaffold for Experiment 1 on the formal `exp01_working` subset.

## Inputs

- Train manifest: `/home/tmushuru/datasets/alopecia_public/manifests/exp01_working_train.csv`
- Validation manifest: `/home/tmushuru/datasets/alopecia_public/manifests/exp01_working_val.csv`
- Config: `experiments/configs/exp01_centralised_baseline.yaml`

## Training setup

- Model type: early conditional reconstruction scaffold
- Model name: `ldm-256-cross_attention`
- Image size: `256`
- Condition classes: `1..7`
- Epochs: `3`
- Batch size: `4`
- Device: `cuda`

## Why this run matters

This is the first result showing that the Experiment 1 centralised generative
path is operational end to end: configuration loading, conditional batch
construction, GPU training, and stable train/validation reconstruction losses.

## Saved artefacts

- `summary.json`: machine-readable training summary
- `interpretation.md`: concise research interpretation and next-step framing
