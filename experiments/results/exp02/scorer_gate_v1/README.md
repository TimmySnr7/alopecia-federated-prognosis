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
- Validation severity distribution: severity `3`: `2`, severity `4`: `2`,
  severity `5`: `2`, severity `6`: `1`, severity `7`: `2`
- Features: `131` handcrafted image features
- Gate threshold: validation macro F1 `>= 0.40`

Severity classes `1` and `2` are absent from validation. For that reason, this
package reports both macro F1 over all train-observed labels and macro F1 over
validation-present labels.

## Results

| Formulation | Best model | Accuracy | Macro F1 over train labels | Macro F1 over validation-present labels | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| Exact 7-class | `logreg_c0_5_balanced` | `0.556` | `0.424` | `0.593` | pass |
| Ordinal 3-bin | `extra_trees_balanced` | `0.778` | `0.552` | `0.829` | pass |

The exact seven-class scorer clears the `0.40` gate even under the all-train-label
macro F1 calculation (`0.424`), not only under the validation-present subset
calculation.

## Saved Artefacts

- `summary.json`: full scorer-gate output with all model results, predictions,
  confusion matrices, and classification reports.
- `interpretation.md`: concise scientific interpretation and caveats.
