# Experiment 07 Diagnostics v1

This diagnostic pass checks whether the near-perfect Experiment 07 metadata-ridge result is caused by data leakage, target degeneracy, or normalization artifacts.

## Checks Performed

1. Source-case split overlap.
2. Metadata feature list inspection.
3. Parameter variance and unique-value counts.
4. Unnormalized and normalized MAE reporting.
5. Test-target shuffle control.
6. Train-target shuffle control.
7. Predicted-vs-ground-truth plots for each parameter.

## Main Finding

The result is not explained by source-case leakage or direct parameter leakage. The train, validation, and test case IDs are disjoint, and the metadata model receives only:

- severity scalars,
- severity direction terms,
- dataset-key one-hot features,
- label-schema one-hot features.

However, the parameter space is strongly rule-structured:

- `ellipse_cx`, `mask_blur_radius`, and `texture_blur_radius` are constant.
- `ellipse_cy`, `ellipse_rx`, and `ellipse_ry` have only two unique values.
- edit-strength controls are deterministic functions of source severity, target severity, and global proxy constants.

Therefore, the near-zero metadata-ridge MAE is real but not surprising. It reflects the simplicity of the current proxy parameterisation.

## Critical Control Results

For `metadata_ridge`:

- original test MAE: `7.49e-06`
- original test normalized MAE: `3.71e-05`
- test error against shuffled test truth: `0.1344`
- test error after shuffled train targets: `0.0896`

The error increases sharply under shuffling. This rules out a trivial evaluation bug where predictions would remain near-perfect regardless of target assignment.

## Interpretation

Experiment 07 remains a positive computational result, but its claim should be precise:

> The current Experiment 02 proxy control policy is sufficiently low-dimensional and deterministic that it can be automated by a simple metadata-aware controller.

It should not be claimed that the model has learned clinical progression or complex image-dependent control.

## Next Step

Experiment 07 v2 should enrich the proxy parameter space and then test whether image-aware features improve over severity and metadata:

1. introduce case/image-dependent proxy controls;
2. preserve the deterministic renderer;
3. compare severity-only, metadata-aware, and image-aware controllers;
4. report whether image features reduce parameter error or improve QA beyond metadata.
