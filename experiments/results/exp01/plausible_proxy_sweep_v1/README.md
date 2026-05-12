# Plausible Proxy Sweep (v1)

## Purpose

This package records the first Experiment 1 visual-plausibility-first sweep.

Unlike the scorer-guided residual edit runs, this baseline does not optimize
pixels directly against the classifier. It applies a constrained proxy transform:
lower target severities enhance existing short-hair/stubble texture, while
higher target severities suppress dark stubble, smooth local scalp texture, and
lightly shift tone inside a feathered scalp ellipse.

## Inputs

- Source image: `/home/tmushuru/datasets/alopecia_public/raw/unidatapro_men_hair_loss/4/top.png`
- Source severity: `4`
- Frozen scorer: `/home/tmushuru/datasets/alopecia_public/metadata/exp01_working_severity_baseline_v4.ckpt`

## Proxy setup

- Ellipse mask: `[0.50, 0.52, 0.36, 0.45]`
- Mask blur radius: `10`
- Texture blur radius: `3`
- Hair enhancement strength: `1.2`
- Hair suppression strength: `1.5`
- Smoothing strength: `0.35`
- Tone shift strength: `0.03`

## Why this run matters

This is the first result where the visual trend is plausible and the frozen
scorer's expected severity is perfectly monotonic across targets.

The discrete class predictions remain coarse: targets `1-5` are mostly scored
as severity `4`, while targets `6-7` move to severity `7`. However, the expected
severity values increase monotonically from `4.15` to `5.78`, and the visual
edits are small, mask-localized, and morphology-inspired rather than
adversarial.

## Saved artefacts

- `summary.json`: machine-readable target-by-target sweep summary
- `interpretation.md`: concise research interpretation and caveats
