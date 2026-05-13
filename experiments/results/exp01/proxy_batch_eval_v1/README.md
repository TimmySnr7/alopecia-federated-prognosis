# Proxy Batch Evaluation (v1)

## Purpose

This package records the first batch-level visual-plausibility evaluation for
Experiment 1.

The run applies the morphology-inspired proxy sweep to all `top` view samples
in the current `exp01_working_index.csv`, scores every target with the frozen
severity baseline, and writes mask-QA and sweep contact sheets for visual review.

## Inputs

- Manifest: `/home/tmushuru/datasets/alopecia_public/metadata/exp01_working_index.csv`
- View filter: `top`
- Frozen scorer: `/home/tmushuru/datasets/alopecia_public/metadata/exp01_working_severity_baseline_v4.ckpt`
- Device: `cuda`
- Cases evaluated: `15`

## Acceptance Criteria

- Expected-severity monotonic fraction: `>= 0.8`
- Expected-severity span: `>= 0.5`
- Max unmasked mean absolute delta: `<= 0.01`
- Mask area fraction: `0.20` to `0.75`

## Results

- Auto-pass count: `13 / 15`
- Auto-pass rate: `0.867`
- Mean expected-severity monotonic fraction: `0.878`
- Mean expected-severity span: `2.157`

The two automatic failures were both from
`unidpro_hair_loss_male_norwood_scale` top-down images where the proxy/scorer
combination did not produce enough expected-severity separation.

## Contact Sheets

The generated visual QA artefacts are intentionally not tracked in Git because
they are image-heavy outputs:

- `/home/tmushuru/datasets/alopecia_public/metadata/exp01_proxy_batch_eval_v1/mask_qa_contact_sheet.png`
- `/home/tmushuru/datasets/alopecia_public/metadata/exp01_proxy_batch_eval_v1/proxy_sweep_contact_sheet.png`

## Saved Artefacts

- `summary.json`: full machine-readable batch summary with per-case target scores
- `cases.csv`: compact case-level table for spreadsheet review
- `interpretation.md`: concise research interpretation and next bottleneck
