# Experiment 01 Final Report

## Status

Experiment 01 is complete as a feasibility and pipeline-validation experiment.

It should not be interpreted as a solved clinical prognosis-generation system.
The experiment successfully established an auditable data pipeline, severity
baseline, target-control baseline, visual-plausibility proxy, and batch-level QA
process on the current public top-view subset.

## Research Question

Can the project build a reproducible first-pass pipeline for severity-conditioned
alopecia image editing using the available public data?

The answer is yes, with important scope limits.

Experiment 01 demonstrates that target-directed severity editing is measurable
and that visually plausible severity-direction changes can be produced under
explicit constraints. It also shows that naive learned generative scaffolds and
unconstrained scorer-guided optimization are not sufficient.

## Data Scope

The final working dataset used the `exp01_working` manifests derived from the
curated public pool:

- Working train samples: `41`
- Working validation samples: `9`
- Working test samples: `7`
- Top-view samples used for batch plausibility QA: `15`
- Severity classes observed: `1-7`

The top-view subset remains small and heterogeneous. Results should therefore be
read as feasibility evidence rather than population-level validation.

The 15 top-view cases used for batch proxy QA were balanced by source dataset
and source severity, but not by acquisition conditions:

| Characteristic | Distribution |
| --- | --- |
| Dataset source | `unidatapro_women_hair_loss`: 5; `unidatapro_men_hair_loss`: 5; `unidpro_hair_loss_male_norwood_scale`: 5 |
| Source severity | `1`: 3; `2`: 3; `3`: 3; `4`: 3; `5`: 3 |
| Split | train: 9; validation: 2; test: 4 |
| View | all labelled `top` |
| Acquisition heterogeneity | variable resolution, lighting, rotation, camera distance, background, and head centering |
| Clinical-label heterogeneity | mixed `norwood_stage` and `ludwig_stage` proxy labels from public sources |

Combining Norwood and Ludwig labels into one ordinal severity proxy is a
methodological simplification. Experiment 01 treats them as a shared severity
axis for feasibility testing, but this assumption requires validation or
replacement in Experiment 02. A future scorer trained on a conflated
Norwood/Ludwig axis may learn label-source noise rather than a robust alopecia
severity signal.

## Completed Milestones

### 1. Manifest and Loader Pipeline

The project can now build and consume manifest CSV files containing image paths,
dataset keys, view labels, severity proxies, and train/validation/test splits.

Smoke tests confirmed that the loader can produce image tensors and severity
labels for both full working manifests and top-view subsets.

### 2. Severity Baseline

The working severity baseline was trained with pretrained weights,
augmentation, and class weighting.

Key result from `severity_baseline_working_v3`:

- Train samples: `41`
- Validation samples: `9`
- Final train accuracy: `0.512`
- Final validation accuracy: `0.222`
- Validation macro F1: `0.15`

This classifier is useful as a weak experimental scorer, but it is not strong
enough to be a sole clinical or scientific evaluator.

For context, a uniform random classifier over seven classes would have expected
accuracy of approximately `0.143`, and macro F1 is also approximately `0.143`
under a balanced no-skill setting. The observed validation accuracy is therefore
only marginally above a simple random baseline, while macro F1 is essentially at
chance level. The working training distribution is imbalanced, with class `7`
over-represented, so accuracy is especially fragile; macro F1 is the more
appropriate warning signal here.

### 3. Learned Generative Scaffolds

Several centralised conditional generative scaffolds were trained and evaluated.

The key outcome is negative but useful: the learned scaffolds optimized
reconstruction-like objectives but did not produce target-separated,
visually meaningful severity sweeps. Later variants using frozen-scorer guidance
also collapsed visually.

Documented scaffold milestones include:

- `generative_scaffold_working_v1`
- `generative_scaffold_working_v8`
- `generative_scaffold_working_v9`
- `generative_scaffold_working_v10`

These runs establish that the current small dataset and scaffold design are not
enough for naturalistic learned prognosis-style generation.

### 4. Scorer-Guided Target Control

`guided_edit_baseline_v2` showed that explicit scorer-guided residual editing
can achieve strong target control.

Key result:

- Expected-severity monotonic fraction: `1.0`
- Predicted-severity monotonic fraction: `1.0`

This was the first clear positive control for target conditioning. However, the
outputs were visually stylized and adversarial, so the run supports
controllability but not visual plausibility.

The adversarial artefacts were primarily visual texture and colour shortcuts
that changed the scorer response without producing clinically plausible
alopecia morphology. Earlier unconstrained and weakly constrained sweeps showed
global colour casts, patchy scalp texture, boundary noise, and checker/banding
artefacts inside the editable region, especially at larger target shifts away
from the source severity. These artefacts are not part of the 13/15
batch-QA-passing cases, which used the morphology-inspired proxy rather than
direct scorer-guided residual optimization.

### 5. Visual Plausibility Proxy

`plausible_proxy_sweep_v1` introduced a morphology-inspired proxy edit:

- lower severity targets emphasize existing short-hair/stubble texture,
- higher severity targets suppress dark stubble and smooth local scalp texture,
- edits are constrained inside a feathered scalp ellipse.

Key single-image result:

- Expected-severity monotonic fraction: `1.0`
- Predicted-severity monotonic fraction: `1.0`
- Visual edits were small, local, and substantially more plausible than
  adversarial scorer-guided edits.

This is not a learned generator, but it is the first usable visual plausibility
guardrail for the project.

The single-image proxy result achieved perfect monotonicity because it used a
well-centered top-view bald/scalp image for which the fixed ellipse was a
reasonable edit mask. This should be read as a positive control rather than as
evidence of generalization.

### 6. Batch-Level Proxy QA

`proxy_batch_eval_v1` scaled the visual-plausibility proxy to all current
top-view samples and generated both numeric metrics and contact-sheet QA.

Key result:

- Cases evaluated: `15`
- Auto-pass count: `13`
- Auto-pass rate: `0.867`
- Mean expected-severity monotonic fraction: `0.878`
- Mean expected-severity span: `2.157`

The automatic pass criteria checked monotonicity, expected-severity span,
unmasked visual delta, and mask area.

The contact sheets revealed the main remaining bottleneck: the fixed ellipse
mask is useful for first-pass QA, but it is not source-aligned for every image.

The automatic pass criteria were:

| Criterion | Threshold |
| --- | --- |
| Expected-severity monotonic fraction | `>= 0.8` |
| Expected-severity span | `>= 0.5` |
| Maximum unmasked mean absolute delta | `<= 0.01` |
| Mask area fraction | `0.20` to `0.75` |

These thresholds were exploratory engineering criteria chosen to make the first
batch QA auditable, not pre-registered statistical success criteria. The `0.867`
pass rate should therefore be interpreted as exploratory feasibility evidence.

The batch mean expected-severity monotonic fraction (`0.878`) is lower than the
single-image proxy result (`1.0`) because source alignment and scorer response
vary across the 15 top-view cases. The two automatic failures were also the two
cases with the lowest expected-severity monotonic fractions:

- `unidpro_hair_loss_male_norwood_scale`, source severity `5`:
  monotonic fraction `0.333`, expected-severity span `0.153`
- `unidpro_hair_loss_male_norwood_scale`, source severity `2`:
  monotonic fraction `0.000`, expected-severity span `0.403`

Both automatic failures came from `unidpro_hair_loss_male_norwood_scale`. With
only five cases from that source, this should not be over-interpreted as a
statistical dataset effect. However, it is a useful warning that source-specific
acquisition conditions, head centering, or label conventions may be less
compatible with the fixed ellipse proxy than the two `unidatapro` sources.

## Main Findings

1. The data and manifest pipeline is functional and reproducible.
2. A weak severity scorer can be trained and used for experimental feedback.
3. Learned generative scaffolds currently fail to produce meaningful visual
   target separation.
4. Direct scorer optimization can create target control but is vulnerable to
   adversarial visual artifacts.
5. A constrained morphology-inspired proxy can produce more plausible
   severity-direction changes.
6. Batch-level QA is now in place and shows promising but imperfect results.
7. Robust scalp/head localization is now the primary engineering bottleneck.

## What Experiment 01 Supports

Experiment 01 supports claiming that the project has a working, auditable
feasibility pipeline for severity-conditioned alopecia image editing on the
current public-data subset.

It also supports using the plausible proxy outputs as a baseline, sanity check,
and possible source of pseudo-paired supervision for later learned models.

Proxy outputs should only be used as pseudo-paired supervision after mask QA.
Cases with visibly misaligned masks, excessive background editing, or failed
automatic plausibility criteria should be excluded from any learned model's
training set. This gating is required to avoid teaching a downstream model
spurious mask-boundary or background artefacts.

## What Experiment 01 Does Not Support

Experiment 01 does not support claiming:

- clinically valid prognosis generation,
- patient-specific longitudinal prediction,
- robust generalization across datasets,
- or a production-quality learned generative model.

The severity labels are proxies, the dataset is small, and the scorer is too weak
to function as the only evaluator.

## Recommended Experiment 02

Experiment 02 should focus on robust visual plausibility and source alignment.

Recommended goals:

1. Add source-specific scalp/head localization.
2. Add a mask-review or mask-quality gate before exporting proxy pairs.
3. Generate a curated pseudo-paired training set from visually valid proxy edits.
4. Train a learned residual editor against the proxy pairs.
5. Evaluate learned outputs using both scorer metrics and contact-sheet visual QA.

Experiment 02 should not proceed to learned residual-editor training until a
stronger severity scorer is available. A suggested hard entry criterion is a
validation macro F1 of at least `0.40` on the working validation set, or an
equivalently justified ordinal metric if the classifier is reformulated as an
ordinal severity model. Without a stronger scorer, predicted-severity
monotonicity is not scientifically interpretable.

Experiment 02 should only claim improvement if it beats the proxy baseline on:

- visual plausibility,
- expected-severity monotonicity,
- target separation,
- low background/unmasked drift,
- and source identity preservation.

The Experiment 01 proxy baseline values to carry into the Experiment 02 planning
document are:

| Metric | Experiment 01 proxy baseline |
| --- | --- |
| Mean expected-severity monotonic fraction | `0.878` |
| Mean expected-severity span | `2.157` |
| Auto-pass rate under exploratory QA criteria | `0.867` |
| Maximum unmasked mean absolute delta criterion | `<= 0.01` |

Before Experiment 02 begins, these qualitative improvement dimensions should be
operationalised. A defensible starting point is to require mean expected-severity
monotonic fraction `>= 0.95`, no reduction in visual plausibility under
contact-sheet review, no increase in unmasked/background drift beyond the
Experiment 01 threshold, and explicit source-identity preservation assessment
using either a second human rater, a perceptual similarity metric, or both.

## Final Conclusion

Experiment 01 is successful as a proof-of-pipeline and feasibility baseline.

The project should now move from asking "can we make the pipeline work?" to
asking "can we learn visually plausible, source-aligned edits that outperform the
hand-designed proxy?"
