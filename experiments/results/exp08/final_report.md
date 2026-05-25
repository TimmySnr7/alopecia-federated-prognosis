# Experiment 08 Final Report

## Title

QA-Constrained Hybrid Refinement for Image-Aware Proxy-Parameter Control

## Status

Completed on AI5. Suitable for milestone closure as a positive computational result.

## Research Question

Can the Experiment 07 v2 trade-off between image-aware parameter accuracy and severity-order robustness be eliminated using sparse image-aware control and bounded QA-constrained refinement?

## Method

Experiment 08 reused the Experiment 07 v2 proxy-parameter controller pipeline and evaluated six variants:

1. `sparse2_ridge`: ridge regression using hair-presence index and masked grayscale mean.
2. `sparse4_ridge`: ridge regression using hair-presence index, masked grayscale mean, mask area fraction, and texture p90.
3. `sparse4_refined_lambda_0_5`: bounded refinement from sparse-4 with monotonicity weight 0.5.
4. `sparse4_refined_lambda_1_0`: bounded refinement from sparse-4 with monotonicity weight 1.0.
5. `sparse4_refined_lambda_5_0`: bounded refinement from sparse-4 with monotonicity weight 5.0.
6. `sparse4_refined_lambda_10_0`: bounded refinement from sparse-4 with monotonicity weight 10.0.

All variants were evaluated on the same two validation and four test source cases used in Experiments 03-07. The rebuilt neural severity scorer from Experiment 05 was used for expected-severity metrics.

## Results

### Direct Comparison to Experiment 07 v2

| Model | Feature set | Test proxy / parameter error | Test monotonicity | Test violations | Interpretation |
|---|---|---:|---:|---:|---|
| Exp07 v2 image-stats ridge | 9 global image statistics | 0.0084 parameter MAE | 0.958 | 1 | Lowest Exp07 v2 parameter error, but introduced one severity-order inversion |
| Exp08 sparse-2 ridge | Hair-presence index + masked grayscale mean | 0.000289 proxy L1 | 1.000 | 0 | Most parsimonious successful controller |
| Exp08 sparse-4 ridge | Sparse-2 + mask area fraction + texture p90 | 0.000281 proxy L1 | 1.000 | 0 | Slightly lower proxy L1 with two additional features |

This comparison shows that the Experiment 07 v2 monotonicity inversion was not caused by image-aware control itself. It was caused by feature over-inclusion in the nine-statistic representation. Sparse image-aware features restore severity-order robustness while preserving proxy fidelity.

### Experiment 08 Variants

| Variant | Split | Auto-pass | Mono. fraction | Violations | Spearman rho | Span | Drift | Proxy L1 | Refinement success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sparse-2 ridge | Val | 1.000 | 1.000 | 0.000 | 1.000 | 4.328 | 0.0020 | 0.000215 | - |
| Sparse-2 ridge | Test | 1.000 | 1.000 | 0.000 | 1.000 | 5.360 | 0.0060 | 0.000289 | - |
| Sparse-4 ridge | Val | 1.000 | 1.000 | 0.000 | 1.000 | 4.328 | 0.0020 | 0.000265 | - |
| Sparse-4 ridge | Test | 1.000 | 1.000 | 0.000 | 1.000 | 5.360 | 0.0060 | 0.000281 | - |
| Sparse-4 refined, lambda 0.5 | Val | 1.000 | 1.000 | 0.000 | 1.000 | 4.327 | 0.0021 | 0.000427 | 1.000 |
| Sparse-4 refined, lambda 0.5 | Test | 1.000 | 1.000 | 0.000 | 1.000 | 5.359 | 0.0061 | 0.000785 | 1.000 |
| Sparse-4 refined, lambda 1.0 | Val | 1.000 | 1.000 | 0.000 | 1.000 | 4.328 | 0.0021 | 0.000402 | 1.000 |
| Sparse-4 refined, lambda 1.0 | Test | 1.000 | 1.000 | 0.000 | 1.000 | 5.360 | 0.0060 | 0.000575 | 1.000 |
| Sparse-4 refined, lambda 5.0 | Val | 1.000 | 1.000 | 0.000 | 1.000 | 4.328 | 0.0021 | 0.000374 | 1.000 |
| Sparse-4 refined, lambda 5.0 | Test | 1.000 | 1.000 | 0.000 | 1.000 | 5.359 | 0.0061 | 0.000601 | 1.000 |
| Sparse-4 refined, lambda 10.0 | Val | 1.000 | 1.000 | 0.000 | 1.000 | 4.328 | 0.0021 | 0.000349 | 1.000 |
| Sparse-4 refined, lambda 10.0 | Test | 1.000 | 1.000 | 0.000 | 1.000 | 5.359 | 0.0061 | 0.000693 | 1.000 |

Refinement success rate measures the fraction of held-out cases where post-prediction optimisation preserved the QA target of monotonicity >= 0.8 and drift <= 0.01 within the configured iteration budget. All refined variants achieved a success rate of 1.000, confirming that the optimisation procedure was numerically stable.

## Success Criteria

| Criterion | Result | Assessment |
|---|---|---|
| Restore monotonicity to 1.000 | Sparse-2, sparse-4, and all refined variants achieved 1.000 on validation and test | Passed |
| Eliminate monotonicity violations | All variants produced zero mean violations on validation and test | Passed |
| Maintain 100% auto-pass | All variants achieved 1.000 auto-pass on validation and test | Passed |
| Keep unmasked drift <= 0.01 | All variants remained below 0.0061 on test | Passed |
| Preserve image-aware parameter fidelity | Sparse-4 ridge achieved test proxy L1 of 0.000281; sparse-2 achieved 0.000289 | Passed |
| Show refinement improves the controller | Refinement preserved QA but increased proxy L1 relative to unrefined sparse ridge | Not needed / not supported |

## Interpretation

Experiment 08 resolves the Experiment 07 v2 monotonicity trade-off. The result is stronger than expected because the sparse feature controllers themselves recover perfect severity-order robustness. This indicates that the earlier image-stat inversion was not caused by image-aware control in general, but by using a larger global-statistic feature set that introduced mild cross-case noise.

The sparse feature set removes correlated global statistics, such as dark-pixel fraction, bright-pixel fraction, unmasked grayscale mean, and texture mean, that can capture cross-case lighting and exposure differences rather than hair-specific progression signals. Retaining hair-presence index and masked grayscale mean isolates the most severity-relevant appearance information while reducing dataset-specific noise.

The bounded QA-refinement variants are useful as a stress test. They show that post-prediction refinement can maintain pass validity and avoid monotonicity failures. However, because the sparse unrefined controllers already satisfy all QA criteria and have lower proxy L1, refinement is not the recommended production path at this stage.

The recommended production controller is `sparse2_ridge`, using hair-presence index and masked grayscale mean. It achieves identical QA performance to `sparse4_ridge` with perfect monotonicity, 100% auto-pass, and test drift of approximately 0.0060, while using only two interpretable features. The proxy-L1 difference between sparse-2 and sparse-4 is negligible at this data scale. Both avoid the direct-generation failure modes observed in Experiments 03-06 because image synthesis remains deterministic and proxy-rendered.

## Contact-Sheet Review

The test contact sheets for `sparse2_ridge` and `sparse4_ridge` show source-preserving, scalp-localised severity sweeps without non-scalp hallucination. The visible progression is monotonic across all four held-out test cases. This should be carried forward into a structured face-validity pilot using a seven-point rating scale for source preservation, scalp localization, monotonic progression, and anatomical plausibility.

## Limitations

The experiment remains proxy-supervised. It demonstrates reliable learned control over the validated proxy renderer, not clinical validity of the renderer itself. The held-out set is still small, with two validation and four test source cases. Finally, the current QA gate is now saturated for successful controller variants; future evaluation should add a stricter perceptual or human-rated criterion to rank controllers that already pass computational QA.

## Conclusion

Experiment 08 is a positive result. Sparse image-aware proxy-parameter control restores perfect held-out monotonicity while preserving low drift and 100% auto-pass. QA-constrained refinement is feasible but unnecessary under the current renderer. The recommended next phase is clinical or near-clinical face-validity testing of `sparse2_ridge` outputs, while keeping the proxy-supervised limitation explicit.
