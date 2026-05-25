# Experiment 03 Final Report

## Status

Experiment 03 is complete as a negative learned residual-editor experiment.

It should not be interpreted as a successful learned prognosis-generation
model. Its useful contribution is showing that the current tiny QA-gated
Experiment 02 proxy-pair pool is not enough for this residual editor to learn
reliable target-conditioned severity separation.

## Research Question

Can a small mask-constrained residual editor learn the Experiment 02
QA-gated proxy edits while preserving source identity and visual plausibility?

The answer is no under the current data and model constraints.

The editor can keep edits visually plausible and mostly mask-localized, but it
collapses target-severity separation. A second run that explicitly countered
under-conditioning did not fix the problem.

## Data Scope

Experiment 03 used the Experiment 02 QA-gated proxy-pair manifest:

- Accepted source cases: `13`
- Severity-shift proxy-pair rows: `78`
- Training pairs: `42`
- Validation pairs: `12`
- Test pairs: `24`
- Training cases: `7`
- Validation cases: `2`
- Test cases: `4`

The Experiment 03 manifest contains only the `13` Experiment 02 QA-passing
source cases. The two Experiment 02 proxy-QA failures were excluded before this
experiment and therefore do not appear in any split. This is why the proxy
baseline pass rate is `1.000` in the Experiment 03 validation and test sweeps:
the proxy is being re-evaluated only on its own QA-gated subset.

Splitting was performed at the source-case level, not at the individual
proxy-pair row level. No source case or source image path appears in more than
one of the training, validation, or test splits.

The supervision pool is very small and heterogeneous. Results should be read as
pipeline and feasibility evidence, not as a population-level model comparison.

## Completed Runs

### v1: Baseline Residual Editor

`proxy_residual_editor_v1` trained a small mask-constrained editor with texture
channels and reconstruction-style losses.

| Split | Learned pass rate | Proxy pass rate | Learned mean span | Proxy mean span | Mean learned max unmasked delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | `0.000` | `1.000` | `0.291` | `1.816` | `0.00022` |
| Test | `0.250` | `1.000` | `0.301` | `2.276` | `0.00175` |

The model preserved visual plausibility and localization, but under-responded
to target severity.

### v2: Under-conditioning Countermeasure

`proxy_residual_editor_v2` added severity-delta weighting, dataset-balanced
sampling, stronger masked residual reconstruction, increased residual scale,
and scorer-space consistency losses.

| Split | Learned pass rate | Proxy pass rate | Learned mean span | Proxy mean span | Mean learned max unmasked delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | `0.000` | `1.000` | `0.300` | `1.816` | `0.00199` |
| Test | `0.000` | `1.000` | `0.043` | `2.276` | `0.00325` |

The countermeasures did not recover target separation. Test expected-severity
span became worse than v1, and auto-pass dropped to `0 / 4` on test.
The most likely mechanism is that the stronger weighted and scorer-space losses
overfit the tiny two-case validation signal and encouraged a source/proxy
residual template that did not generalise to the four held-out test cases,
rather than learning a robust target-conditioned edit direction.

## Main Findings

1. The proxy baseline remains clearly stronger than the learned editor for
   target-severity separation.
2. The residual editor can preserve visual plausibility better than the
   earlier Experiment 01 learned scaffolds.
3. The dominant failure is not background drift or gross visual collapse; it is
   target-conditioned severity-span collapse.
4. Direct low-complexity fixes for under-conditioning were insufficient.
5. Experiment 03 does not support using the current learned residual editor as
   the project's final prognosis-generation model.

## What Experiment 03 Supports

Experiment 03 supports keeping the Experiment 02 proxy pipeline as the current
auditable baseline for target-controlled visual severity sweeps.

It also supports the claim that learned editing needs either more data, a
different model formulation, or a stronger pretrained editing backbone before it
can replace the proxy baseline.

## What Experiment 03 Does Not Support

Experiment 03 does not support claiming:

- a clinically valid learned prognosis generator,
- a learned editor that beats the Experiment 02 proxy baseline,
- robust target-conditioned image editing from the current `78` proxy-pair
  supervision rows,
- or that additional scalar loss-weight tuning is likely to solve the core
  failure mode.

## Recommended Next Experiment

The next learned-editing attempt should be a new experiment rather than another
Experiment 03 tuning run.

The primary recommended direction is a pretrained image-editing backbone with
mask-localized adaptation, because this is the option most directly aligned
with the proposal's Contribution C1 latent conditional diffusion architecture
and with the Experiment 02 supervisory recommendation to pivot to the latent
diffusion stack. Secondary directions that may support that pivot include:

- expanding the QA-passing pseudo-pair pool before learning,
- predicting normalized proxy residuals with explicit source and target
  endpoints,
- adding target-contrastive losses across same-source severity sweeps,
- and using the QA-gated proxy pairs as weak supervision for the pretrained
  latent editing model.

## Final Conclusion

Experiment 03 is a useful negative result.

It shows that the project can train and audit a learned residual editor, but
the current small-data residual formulation cannot preserve the Experiment 02
proxy baseline's target-severity separation. The final basis for the project is
therefore: Experiment 02 remains the proxy-supervision and QA baseline, while
Experiment 03 closes the current learned residual-editor path as insufficient.
