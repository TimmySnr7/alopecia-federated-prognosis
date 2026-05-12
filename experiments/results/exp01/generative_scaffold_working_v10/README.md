# Generative Scaffold: Working Subset (v10)

## Purpose

This package records the first Experiment 1 run that rebalanced frozen external
severity guidance against a weak shifted-target image-preservation term.

## Inputs

- Train manifest: `/home/tmushuru/datasets/alopecia_public/manifests/exp01_working_train.csv`
- Validation manifest: `/home/tmushuru/datasets/alopecia_public/manifests/exp01_working_val.csv`
- Config: `experiments/configs/exp01_centralised_baseline.yaml`
- Frozen scorer: `/home/tmushuru/datasets/alopecia_public/metadata/exp01_working_severity_baseline_v4.ckpt`

## Training setup

- Model type: conditional reconstruction scaffold with frozen external severity guidance
- Model name: `ldm-256-cross_attention`
- Image size: `256`
- Condition classes: `1..7`
- Epochs: `15`
- Batch size: `4`
- Severity loss weight: `0.3`
- Shifted reconstruction weight: `0.15`
- Train target mode: `sampled`
- Train target sampling strategy: `adjacent`
- Train target shift probability: `0.8`
- Maximum target delta: `1`
- Device: `cuda`

## Why this run matters

This run tests whether gentler guidance and a small shifted-target consistency
term can preserve image structure better than `v9`.

## Saved artefacts

- `summary.json`: machine-readable training summary
- `interpretation.md`: concise research interpretation and why `v11` follows
