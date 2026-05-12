# Generative Scaffold: Working Subset (v9)

## Purpose

This package records the first Experiment 1 run that combined:

- external frozen severity guidance,
- adjacent target-shift sampling,
- and the formal `exp01_working` subset.

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
- Severity loss weight: `1.0`
- Train target mode: `sampled`
- Train target sampling strategy: `adjacent`
- Train target shift probability: `0.8`
- Maximum target delta: `1`
- Device: `cuda`

## Why this run matters

This is the first generative run where adjacent severity transitions proved more
learnable than full-range random target jumps under frozen external guidance.

## Saved artefacts

- `summary.json`: machine-readable training summary
- `interpretation.md`: concise research interpretation and why `v10` follows
