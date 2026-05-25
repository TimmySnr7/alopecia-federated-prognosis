# Experiment 07 v2 Interpretation

## Verdict

Experiment 07 v2 is a positive but bounded result. Enriching the proxy parameter space with source-image statistics makes image-aware control useful in parameter space, but not yet necessary for rendered QA success.

The image-statistics ridge controller achieves the lowest absolute parameter MAE on both validation and test. However, metadata-ridge remains the cleaner rendered model because it preserves perfect monotonicity and also passes all QA gates. The practical result is that the current deterministic proxy renderer is still adequately controlled by severity and metadata alone. The dominant evaluation limitation is therefore QA-gate saturation: the gate rejects failed models but cannot rank controllers once they all pass.

## Quantitative Summary

| Model | Val MAE | Test MAE | Val auto-pass | Test auto-pass | Test monotonicity | Test span | Test drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Constant mean | 0.1273 | 0.1032 | 0.000 | 0.000 | 1.000 | 0.000 | 0.0020 |
| Severity ridge | 0.0535 | 0.0167 | 1.000 | 1.000 | 1.000 | 5.350 | 0.0053 |
| Metadata ridge | 0.0190 | 0.0123 | 1.000 | 1.000 | 1.000 | 5.352 | 0.0059 |
| Image-stats ridge | 0.0110 | 0.0084 | 1.000 | 1.000 | 0.958 | 5.359 | 0.0053 |

## What Changed From v1

Experiment 07 v1 showed that the original proxy-control vector was almost completely deterministic from severity and metadata. V2 introduced image-statistic-dependent edit multipliers so that visual information could matter.

This worked in parameter space: image-stats ridge reduced test parameter MAE from `0.0123` for metadata-ridge to `0.0084`. The improvement is real but small. It did not produce a superior image-level result because both models already pass the full QA gate.

## Scientific Meaning

The main scientific conclusion is that the learned-controller path is viable and robust. Unlike the direct diffusion experiments, it does not hallucinate because final image formation remains deterministic and mask-constrained. The learned component controls a transparent parameter vector.

The secondary conclusion is that the current proxy parameterisation is still too simple. Even when image statistics are introduced, the renderer is forgiving enough that metadata-ridge already achieves full held-out QA success.

## Failure / Weakness

The image-aware model introduced one mild monotonicity inversion in the test set, reducing mean test monotonicity to `0.958`. It still passed the auto-pass gate because the criterion permits at most one local inversion in a seven-step sweep. This suggests that image-aware control should include an explicit monotonicity constraint or post-hoc QA refinement if developed further.

The monotonicity violation also shows that minimizing parameter-space error does not perfectly align with preserving severity-order constraints. Global image statistics may reduce parameter error by capturing appearance cues that are useful on average but not strictly severity-order preserving on every held-out case. A likely production formulation is therefore hybrid: image-aware parameter prediction followed by explicit monotonicity regularization or QA-constrained parameter refinement.

## Recommendation

Close Experiment 07 as a positive controller result, with v2 as an ablation showing that image-aware control improves parameter error but not current QA outcomes. The next phase should not be another linear-controller variant. It should either:

1. expand the proxy renderer with clinically motivated, image-dependent controls, or
2. add QA-constrained parameter optimization on top of the metadata controller.

The highest-value next step is a QA-refined proxy controller: predict parameters with metadata/image features, render, then optimize parameters under monotonicity, span, drift, and mask constraints.

## Image-Statistic Ablation

A follow-up ablation found that the full nine-statistic image-aware model is not the best parameter predictor. The useful signal is concentrated in a small subset. Hair-presence index alone reduces test MAE to `0.0023`; adding masked gray mean reduces it to `0.0022` and normalized MAE to `0.0156`. Adding mask-area fraction and masked texture 90th percentile gives the best normalized MAE (`0.0148`). In contrast, using all nine image statistics produces test MAE `0.0084` and normalized MAE `0.1240`.

This suggests that sparse, interpretable image features are preferable to a larger hand-crafted feature set at the current data scale. Extra correlated statistics appear to overfit or destabilize the ridge model.

The remaining feature limitation is spatial. The current statistics summarize the masked scalp globally and do not represent local follicle distribution, directional density gradients, or edge sharpness. Those richer features may be useful in a later controller, but they should be introduced with monotonicity-aware validation rather than as unconstrained feature expansion.
