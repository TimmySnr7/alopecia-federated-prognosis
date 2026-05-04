# Severity Baseline: Working Subset (v3)

## Purpose

This package records the stronger severity-classification baseline on the formal
`exp01_working` subset after adding richer validation reporting.

## Inputs

- Train manifest: `/home/tmushuru/datasets/alopecia_public/manifests/exp01_working_train.csv`
- Validation manifest: `/home/tmushuru/datasets/alopecia_public/manifests/exp01_working_val.csv`

## Training setup

- Model: `resnet18`
- Pretrained backbone: `true`
- Augmentation: `true`
- Class weighting: `true`
- Epochs: `10`
- Batch size: `4`
- Device: `cuda`

## Why this run matters

This run confirms that the formal `exp01_working` subset is the right
development subset for the next phase. It also provides richer class-level
diagnostics beyond plain validation accuracy, making it the final classification
baseline checkpoint before the first centralised generative scaffold.

## Saved artefacts

- `summary.json`: machine-readable training and evaluation summary
- `interpretation.md`: concise analysis and limitations
