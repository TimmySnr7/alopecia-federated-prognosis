# Guided Edit Baseline (v2)

## Purpose

This package records the first Experiment 1 baseline that achieved meaningful,
auditable target-severity control on the current public subset.

Unlike the learned centralised scaffolds, this baseline performs explicit
per-image residual optimization under a frozen severity classifier.

## Inputs

- Source image: `/home/tmushuru/datasets/alopecia_public/raw/unidatapro_men_hair_loss/4/top.png`
- Frozen scorer: `/home/tmushuru/datasets/alopecia_public/metadata/exp01_working_severity_baseline_v4.ckpt`

## Optimization setup

- Method: scorer-guided residual editing
- Steps: `150`
- Learning rate: `0.03`
- Residual scale: `0.08`
- L2 weight: `0.05`
- TV weight: `0.10`
- Device: `cuda`

## Why this run matters

This is the first result in `exp01` where target severity control is clearly
measurable rather than merely hoped for. The frozen scorer predicts the intended
severity class for every target in the sweep, and the expected severity values
are strictly monotonic.

## Saved artefacts

- `summary.json`: machine-readable target-by-target sweep summary
- `interpretation.md`: concise research interpretation and caveats
