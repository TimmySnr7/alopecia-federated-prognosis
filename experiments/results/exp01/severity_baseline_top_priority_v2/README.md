# Severity Baseline: Top-Priority Subset (v2)

## Purpose

This package records the first stable severity-classification baseline judged
strong enough to formalise as an Experiment 1 result.

## Inputs

- Train manifest: `/home/tmushuru/datasets/alopecia_public/manifests/exp01_top_priority_train.csv`
- Validation manifest: `/home/tmushuru/datasets/alopecia_public/manifests/exp01_top_priority_val.csv`

## Training setup

- Model: `resnet18`
- Pretrained backbone: `true`
- Augmentation: `true`
- Class weighting: `true`
- Epochs: `10`
- Batch size: `4`
- Device: `cuda`

## Why this run matters

Compared with the earlier `top_only_v1` and `top_priority_v1` runs, this setup
produced the most stable validation behaviour on the currently available public
data. It is still a feasibility baseline rather than a benchmark-quality result,
but it provides the first defensible end-to-end proof that the Experiment 1
pipeline is trainable on GPU with usable severity-conditioned signal.

## Saved artefacts

- `summary.json`: machine-readable training summary
- `interpretation.md`: concise research interpretation and caveats
