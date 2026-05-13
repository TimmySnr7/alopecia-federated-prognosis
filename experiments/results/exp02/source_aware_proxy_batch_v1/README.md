# Source-Aware Proxy Batch Evaluation (v1)

## Purpose

This run tests whether Experiment 01's fixed ellipse proxy can be made more
source-aligned by using dataset-specific mask presets.

It also exports a QA-gated proxy-pair manifest that can be used as pseudo-paired
supervision input for later learned residual editing experiments.

## Inputs

- Manifest: `/home/tmushuru/datasets/alopecia_public/metadata/exp01_working_index.csv`
- View filter: `top`
- Frozen scorer: `/home/tmushuru/datasets/alopecia_public/metadata/exp01_working_severity_baseline_v4.ckpt`
- Mask presets: `experiments/configs/exp02_mask_presets.json`
- Cases evaluated: `15`

## Acceptance Criteria

| Criterion | Threshold |
| --- | --- |
| Expected-severity monotonic fraction | `>= 0.8` |
| Expected-severity span | `>= 0.5` |
| Maximum unmasked mean absolute delta | `<= 0.01` |
| Mask area fraction | `0.20` to `0.75` |

## Results

- Auto-pass count: `13 / 15`
- Auto-pass rate: `0.867`
- Mean expected-severity monotonic fraction: `0.889`
- Mean expected-severity span: `1.957`
- Source-specific mask uses: `5 / 15`
- Curated severity-shift proxy pairs exported: `78`
- Curated source cases exported: `13`

The mean expected-severity span decreased from the Experiment 01 proxy baseline
of `2.157` to `1.957`. This is treated as an acceptable trade-off because the
source-aware mask preset reduces edit area on the problematic source and makes
the two remaining failures diagnostically cleaner.

## Failure Cases

Both automatic failures came from
`unidpro_hair_loss_male_norwood_scale`, even after using the dataset-specific
mask preset:

| Case | Source severity | Monotonic fraction | Expected span | Reason |
| --- | ---: | ---: | ---: | --- |
| `002_unidpro_hair_loss_male_norwood_scale_5...` | `5` | `0.500` | `0.129` | insufficient scorer/proxy target separation |
| `005_unidpro_hair_loss_male_norwood_scale_2...` | `2` | `0.167` | `0.093` | insufficient scorer/proxy target separation |

The failures had low unmasked drift (`<= 0.0023`) and valid mask area
(`0.246`), so they are treated as scorer/proxy response failures rather than
gross mask-leak failures.

The exported proxy-pair manifest contains six non-identity target severity
shifts per accepted source case (`13 * 6 = 78`). Because only three of five
`unidpro_hair_loss_male_norwood_scale` cases pass QA, the exported manifest is
source-imbalanced and should be used with source-aware sampling or augmentation
in the learned residual-editor experiment.

## Saved Artefacts

- `summary.json`: full batch QA summary with target-level metrics.
- `cases.csv`: compact case-level review table.
- `proxy_pair_manifest.csv`: QA-gated proxy-pair specification for later learned
  residual-editor experiments.
- `interpretation.md`: concise scientific interpretation and next-step decision.

The visual contact sheets were generated at:

- `/home/tmushuru/datasets/alopecia_public/metadata/exp02_proxy_batch_eval_v1/mask_qa_contact_sheet.png`
- `/home/tmushuru/datasets/alopecia_public/metadata/exp02_proxy_batch_eval_v1/proxy_sweep_contact_sheet.png`

Local copies may exist in this directory for inspection, but PNG contact sheets
are intentionally ignored by Git.
