# Experiment 1 Protocol Note

## Purpose

Experiment 1 is the first feasibility-scoped proof of concept for the PhD. It
tests whether a centralised latent diffusion model can generate
severity-conditioned scalp-image scenarios from public alopecia data before
federated learning, privacy mechanisms, or partner data are introduced.

## Claim Boundary

This is a proxy-prognosis experiment, not a true longitudinal forecasting
study. Generated outputs are interpreted as severity-conditioned future-like
scenarios anchored to an ordinal progression proxy rather than verified clinical
future states.

## Pre-Run Audit Gate

The following checks must be completed before the first training run:

1. Confirm dataset licences and whether synthetic sample publication is allowed.
2. Record the label schema used by each included dataset.
3. Verify that the training pool is dominated by top-view scalp imagery.
4. Screen for near-duplicates across pooled datasets.
5. Record a basic note on skin-tone coverage and missingness.

## Calibration Target

ECE, Brier score, and interval coverage are computed on the auxiliary
`severity_class_proxy` head rather than on raw generated pixels. The proxy
target is an ordinal class label harmonised from Norwood/Ludwig-style staging.

## Domain-Adapted FID

The preferred setup uses a ResNet-18 scalp encoder initialised from ImageNet
weights and fine-tuned on held-out scalp images. If that encoder is not ready
at the first run, standard Inception-v3 FID may be used only with an explicit
limitation note in the report.

## Initial Hypotheses

- A model conditioned on current and target severity will generate outputs whose
  derived severity increases as the target severity increases.
- The conditioned model will outperform an unconditioned or random baseline on
  severity-consistency checks and LPIPS.
- Temperature scaling on the auxiliary severity head will reduce ECE on held-out
  data without materially worsening Brier score.
