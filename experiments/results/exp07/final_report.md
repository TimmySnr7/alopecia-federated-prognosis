# Experiment 07 Final Report

## Title

Learned Proxy-Parameter Controller for Alopecia Severity-Sweep Synthesis

## Status

Milestone-ready positive computational result.

## Research Question

Can a learned controller operating in the validated Experiment 02 proxy-parameter space recover target-controlled severity sweeps while avoiding the hallucination and source-preservation failures observed in direct learned image generation?

## Rationale

Experiments 03--06 showed that direct image-generation approaches are not currently viable at the available data scale. Residual editing preserved visual plausibility but collapsed severity span. Zero-shot and adapted latent diffusion variants increased severity response but produced non-scalp hallucinations or failed source preservation. Experiment 07 therefore moved the learned component out of image space and into the low-dimensional proxy-control space.

## Method

Experiment 07 trains lightweight controllers to predict an effective proxy-control vector:

- ellipse mask coordinates,
- mask and texture blur radii,
- severity-scaled hair enhancement,
- severity-scaled tone shift,
- severity-scaled hair suppression,
- severity-scaled smoothing.

The predicted parameters are rendered through the deterministic Experiment 02 proxy renderer, then evaluated with the rebuilt Experiment 05 neural scorer and the same computational QA gate:

- expected-severity monotonicity >= 0.8,
- expected-severity span >= 0.5,
- unmasked drift <= 0.01,
- mask-area fraction between 0.20 and 0.75.

Three controller baselines were evaluated:

1. `constant_mean`: train-set mean parameter vector.
2. `severity_ridge`: source severity, target severity, severity delta, and direction scalars.
3. `metadata_ridge`: severity features plus dataset/schema metadata.

## Results

| Model | Val auto-pass | Test auto-pass | Val monotonicity | Test monotonicity | Val span | Test span | Test unmasked drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Constant mean | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.0020 |
| Severity ridge | 1.000 | 1.000 | 1.000 | 1.000 | 5.651 | 5.346 | 0.0055 |
| Metadata ridge | 1.000 | 1.000 | 1.000 | 1.000 | 4.458 | 5.349 | 0.0061 |

The metadata-aware controller matches the calibrated proxy reference most closely. Its validation and test spans match the proxy reference scale from Experiments 05--06, and all held-out cases pass the auto-pass gate.

## Diagnostic Checks

Because the metadata-aware controller produced near-zero parameter error, an additional diagnostic pass was run. The source-case split is clean: no training source case appears in validation or test. The metadata features do not include source case IDs or proxy parameter values; they include only severity scalars, severity-direction terms, dataset-key indicators, and label-schema indicators.

The near-perfect parameter reconstruction is explained by the current proxy parameterisation rather than by leakage. Several parameters are constant across all rows, the ellipse mask has only two unique presets, and the edit-strength controls are deterministic severity-scaled functions of global proxy constants. Shuffled-target controls confirm this interpretation: the metadata-ridge test MAE increases from `7.49e-06` to `0.1344` when evaluated against shuffled test targets, and to `0.0896` when trained on shuffled target vectors. Therefore, the model is not exploiting an obvious leakage path; it is learning a highly rule-structured target.

## Interpretation

Experiment 07 is a positive result. It demonstrates that the controllable component of the validated proxy mechanism is learnable in a low-dimensional parameter space. This avoids the central failure mode of Experiments 04--06: the learned model cannot hallucinate non-scalp objects because it never generates pixels directly.

The result also clarifies what is currently learnable. Severity features explain the edit-strength controls. Dataset/schema metadata explains source-aware mask-preset differences. This means the present proxy mechanism is highly rule-structured, which is favorable for automation but limits the claim that the model has learned clinical progression.

## Threats to Validity

The strongest limitation is circularity. The controller is trained against parameters derived from the deterministic proxy that it controls. This validates automation of the proxy control policy, not clinical prognosis. The diagnostic pass further shows that the current parameter space is too simple to test whether source-image features add value, because severity and metadata already recover nearly all of the effective proxy-control vector.

The current successful models do not use image features. Therefore, Experiment 07 does not yet prove that source-image content improves parameter selection. An image-aware controller should be tested only after introducing richer proxy parameters or measuring residual error that metadata cannot explain.

The evaluation still depends on the rebuilt neural scorer and the proxy QA gate. Clinical face-validity assessment remains out of scope until the computational system is stable enough to justify expert review.

## Conclusion

Experiment 07 should be treated as the first positive learned result after the Experiment 02 proxy baseline. It confirms that a learned controller over proxy parameters is more tractable than direct latent image generation at the current data scale.

The recommended next experiment is an expanded Experiment 07 v2 with a richer proxy parameter space and an image-aware controller ablation. The key question should be whether image features add value beyond severity and metadata in choosing proxy controls.
