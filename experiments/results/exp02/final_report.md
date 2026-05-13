# Experiment 02 Final Report

## Status

Experiment 02 is complete as a scorer-gated visual plausibility and
proxy-supervision preparation experiment.

It should not be interpreted as a completed learned prognosis-generation model.
The important outcome is narrower and more useful: the project now has an
auditable Experiment 02 gate showing that stronger severity scoring,
source-aware mask QA, and curated proxy-pair export are in place.

## Research Question

Can the project improve the Experiment 01 visual-plausibility pipeline enough to
support QA-gated pseudo-paired supervision for later learned editing?

The answer is yes, with limits.

Experiment 02 establishes that a stronger non-neural severity scorer can clear
the pre-specified entry gate, that source-aware mask presets can be applied and
recorded per case, and that only QA-passing proxy edits can be exported into a
training manifest.

## Data Scope

Experiment 02 used the same current public-data working manifests as Experiment
01:

- Working train samples: `41`
- Working validation samples: `9`
- Validation severity distribution: severity `3`: `2`, severity `4`: `2`,
  severity `5`: `2`, severity `6`: `1`, severity `7`: `2`
- Top-view samples used for source-aware proxy QA: `15`
- Top-view source datasets: `unidatapro_women_hair_loss`,
  `unidatapro_men_hair_loss`, and `unidpro_hair_loss_male_norwood_scale`

The data remain small and heterogeneous. Results should be read as pipeline and
feasibility evidence, not as population-level validation.

Severity classes `1` and `2` are absent from the nine-sample validation set.
This means validation-present macro F1 is computed over severities `3-7`, not
over all seven severities. The exact seven-class model still clears the
Experiment 02 gate under the stricter all-train-label macro F1 calculation
(`0.424 >= 0.40`), but the split is non-standard and should not be treated as a
robust generalization estimate.

## Completed Milestones

### 1. Scorer Entry Gate

Experiment 01 required Experiment 02 to use a stronger severity evaluator before
training learned residual editors.

`scorer_gate_v1` evaluated lightweight feature-based classifiers on the working
train/validation manifests.

Feature-based classifiers were used here because the Experiment 01 neural scorer
remained close to chance on the current tiny validation set, while the immediate
Experiment 02 need was an auditable, fast, small-data engineering gate rather
than an end-to-end neural severity model. This is not a permanent replacement
for neural severity estimation. A neural scorer remains appropriate once the
dataset, augmentation strategy, or pseudo-pair pool is large enough to support
it without overfitting.

Key results:

| Formulation | Best model | Accuracy | Macro F1 over train labels | Macro F1 over validation-present labels | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| Exact 7-class | `logreg_c0_5_balanced` | `0.556` | `0.424` | `0.593` | pass |
| Ordinal 3-bin | `extra_trees_balanced` | `0.778` | `0.552` | `0.829` | pass |

The exact seven-class scorer cleared the pre-specified macro F1 gate of `0.40`.
This is a meaningful improvement over the Experiment 01 neural scorer
(`0.15` macro F1), but remains fragile because validation contains only `9`
samples and excludes severities `1` and `2`.

### 2. Source-Aware Mask QA

Experiment 01 identified fixed ellipse alignment as a bottleneck. Experiment 02
added mask presets through `experiments/configs/exp02_mask_presets.json` and
records the selected mask source in each case summary.

The initial source-aware preset targets the
`unidpro_hair_loss_male_norwood_scale` top-down images, which were the main
failure source in Experiment 01.

Source-aware batch result:

- Cases evaluated: `15`
- Auto-pass count: `13`
- Auto-pass rate: `0.867`
- Mean expected-severity monotonic fraction: `0.889`
- Mean expected-severity span: `1.957`
- Source-specific mask uses: `5`

The headline pass rate stayed the same as Experiment 01, but the diagnostic
quality improved. The two remaining failures now have valid smaller masks and
low unmasked drift, so they are interpreted as insufficient proxy/scorer target
separation rather than gross mask leakage.

The mean expected-severity span decreased from the Experiment 01 proxy baseline
(`2.157`) to `1.957`. This is an acceptable trade-off for this pipeline-hardening
run: the source-aware preset constrains edits more tightly on the
`unidpro_hair_loss_male_norwood_scale` cases, reducing severity response span
but making the remaining failures cleaner and less attributable to background or
mask leakage.

### 3. QA-Gated Proxy-Pair Export

Experiment 02 exported `proxy_pair_manifest.csv` from the source-aware batch
summary.

Only automatic QA-passing cases were exported. No-op identity rows were excluded
from the default manifest.

Exported supervision specification:

- Accepted source cases: `13`
- Severity-shift proxy-pair rows: `78`
- Accepted cases by dataset:
  `unidatapro_women_hair_loss`: `5`,
  `unidatapro_men_hair_loss`: `5`,
  `unidpro_hair_loss_male_norwood_scale`: `3`

The `78` rows come from six non-identity target severities per accepted source
case (`13 * 6 = 78`). The source distribution is imbalanced because two
`unidpro_hair_loss_male_norwood_scale` cases failed QA; `10 / 13` accepted cases
come from the two `unidatapro` sources. This is a potential acquisition-source
confound for the learned residual editor and should be handled with source-aware
sampling, augmentation, or explicit reporting in the next experiment.

The manifest is a specification for reproducible proxy-pair generation, not a
claim that the proxy edits are clinical ground truth.

## Main Findings

1. The Experiment 02 scorer entry criterion is met.
2. Source-aware mask presets are implemented and auditable.
3. The visual proxy remains useful for most top-view cases.
4. The two remaining failed cases should be excluded from pseudo-paired
   supervision until manually corrected.
5. The pseudo-pair export path is now explicit and reproducible.

## What Experiment 02 Supports

Experiment 02 supports using the current source-aware proxy pipeline as a
QA-gated weak-supervision source for the next learned residual-editor experiment.

It also supports using the feature-based scorer as an engineering QA signal for
target separation, provided that visual contact-sheet review remains part of the
evaluation.

## What Experiment 02 Does Not Support

Experiment 02 does not support claiming:

- clinically valid prognosis generation,
- robust generalization across public datasets,
- or that a learned residual editor has already solved visual severity editing.

The pseudo-pair pool is still tiny. It is enough to test the learning pipeline,
but not enough for strong generalization claims.

## Recommended Next Experiment

The next experiment should train a learned residual editor from the
QA-gated `proxy_pair_manifest.csv` and compare against the proxy baseline. The
first learned-editor run should explicitly state whether it trains on the
`78` proxy-pair rows as-is or expands the effective training set with
augmentation. Given the tiny `13`-source-image pool, augmentation and
source-aware sampling should be treated as planned safeguards rather than
optional cosmetic additions.

The learned editor should only be considered an improvement if it:

- preserves source identity and acquisition context better than the proxy,
- matches or exceeds the proxy's expected-severity monotonicity,
- maintains low unmasked/background drift,
- avoids the adversarial colour and texture collapse seen in Experiment 01
  learned scaffolds,
- and passes visual contact-sheet review on held-out top-view cases.

## Final Conclusion

Experiment 02 is successful as a pipeline-hardening step.

It turns Experiment 01's plausibility proxy into an auditable, scorer-gated,
source-aware, QA-gated pseudo-pair supervision pipeline. The project is now ready
for a learned residual-editor experiment, with clear caveats and explicit
failure exclusions.
