# Experiment 02 Scorer Gate (v1)

## Purpose

Experiment 01 ended with a hard entry criterion for Experiment 02: do not proceed
to proxy-pair supervision or learned editing unless a stronger severity evaluator
is available.

This run tests that entry gate using lightweight, auditable feature-based
classifiers on the current working manifests.

## Inputs

- Train manifest: `/home/tmushuru/datasets/alopecia_public/manifests/exp01_working_train.csv`
- Validation manifest: `/home/tmushuru/datasets/alopecia_public/manifests/exp01_working_val.csv`
- Train samples: `41`
- Validation samples: `9`
- Features: `131` handcrafted image features
- Gate threshold: validation macro F1 `>= 0.40`

## Results

| Formulation | Best model | Accuracy | Macro F1 over train labels | Macro F1 over validation-present labels | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| Exact 7-class | `logreg_c0_5_balanced` | `0.556` | `0.424` | `0.593` | pass |
| Ordinal 3-bin | `extra_trees_balanced` | `0.778` | `0.552` | `0.829` | pass |

## Saved Artefacts

- `summary.json`: full scorer-gate output with all model results, predictions,
  confusion matrices, and classification reports.
- `interpretation.md`: concise scientific interpretation and caveats.
