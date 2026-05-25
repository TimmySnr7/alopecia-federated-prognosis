# Experiment 08 v1 Interpretation

## Verdict

Experiment 08 is a positive computational milestone. The central trade-off observed in Experiment 07 v2 was resolved, but not in the expected way: the sparse image-stat controllers themselves restored exact held-out monotonicity. The QA-constrained refinement loop was feasible and preserved pass validity, but it did not improve the practical controller because the unrefined sparse models already satisfied the success criteria.

## Main Finding

Sparse image-aware control is sufficient at the current data scale. The two-feature and four-feature ridge controllers both achieved:

- 100% auto-pass on validation and test.
- Mean monotonic fraction of 1.000 on validation and test.
- Zero monotonicity violations.
- Mean Spearman rho of 1.000.
- Test expected span of approximately 5.36.
- Test unmasked drift below 0.006.

This means the Experiment 07 v2 inversion was not a fundamental property of image-aware control. It was most likely caused by feature over-inclusion or mild cross-case noise in the full nine-statistic model.

## Best Practical Controller

The best practical controller is the unrefined sparse image-stat ridge model.

The sparse-2 controller is the recommended production controller. It uses only hair-presence index and masked grayscale mean, achieves the same QA outcome as sparse-4, and keeps the explanatory burden low for thesis defence and clinical-facing review. The sparse-4 controller is the richer option and adds mask area fraction and texture p90. Both are viable; sparse-4 has slightly lower test proxy L1, while sparse-2 is simpler and marginally stronger on validation proxy L1.

The recommended production candidate for the next face-validity stage is therefore `sparse2_ridge`, with `sparse4_ridge` retained as a secondary sensitivity check.

## Why Sparse Features Worked

The sparse feature set removes correlated global statistics, such as dark-pixel fraction, bright-pixel fraction, unmasked grayscale mean, and texture mean, that can capture cross-case lighting and exposure differences rather than hair-specific progression signals. Retaining hair-presence index and masked grayscale mean isolates severity-relevant appearance information while reducing dataset-specific noise. This explains why the sparse controllers recover perfect monotonicity where the full nine-statistic Experiment 07 v2 controller introduced one inversion.

## Refinement Finding

All bounded refinement variants achieved 100% refinement success rate and preserved perfect test monotonicity. However, refinement increased proxy L1 relative to the unrefined sparse ridge controllers and introduced parameter L2 drift without measurable QA benefit. Under the current renderer, refinement should therefore be treated as a robustness check rather than the production path.

Refinement success rate is defined as the fraction of held-out cases where post-prediction optimisation preserved monotonicity >= 0.8 and drift <= 0.01 within the configured iteration budget.

## Design Implication

Experiment 08 strengthens the controller-based architecture: severity control can be learned in a low-dimensional, interpretable proxy-parameter space while keeping image formation deterministic and hallucination-free by construction. The next clinical-facing step should use the sparse controller output for a structured 2AFC or contact-sheet face-validity pilot, while preserving the current caveat that the renderer remains proxy-supervised rather than clinically validated.
