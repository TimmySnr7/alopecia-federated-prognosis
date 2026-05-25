# Experiment 07 v2 Final Report

## Title

Image-Aware Proxy-Parameter Controller for QA-Gated Alopecia Severity Synthesis

## Status

Completed.

## Research Question

When the proxy parameter space is enriched with source-image statistics, does an image-aware controller improve over severity-only and metadata-aware proxy-parameter control?

## Motivation

Experiment 07 v1 produced a positive result, but diagnostics showed that the original proxy-control vector was highly deterministic. Several parameters were constant, the mask had only two presets, and edit strengths were almost fully determined by source severity, target severity, and metadata. Therefore, v1 did not fairly test whether visual appearance adds useful information.

Experiment 07 v2 addresses this by introducing image-statistic-dependent edit multipliers while preserving the deterministic proxy renderer and QA gate.

## Method

For each source case, masked scalp statistics were extracted:

- masked grayscale mean,
- masked grayscale standard deviation,
- dark-pixel fraction,
- bright-pixel fraction,
- masked texture mean,
- masked texture 90th percentile,
- unmasked grayscale mean,
- mask-area fraction,
- derived hair-presence index.

These statistics modulate the proxy edit controls for hair enhancement, tone shift, hair suppression, and smoothing. Four models were compared:

1. constant mean,
2. severity ridge,
3. metadata ridge,
4. image-stats ridge.

All predictions were rendered on AI5 and evaluated with the Experiment 05 neural scorer and the existing auto-pass gate.

## Results

| Model | Val parameter MAE | Test parameter MAE | Val auto-pass | Test auto-pass | Test monotonicity | Test span | Test unmasked drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp06 best diffusion | N/A | N/A | N/A | 0.000 | 0.500 | 3.555 | 0.0318 |
| Constant mean | 0.1273 | 0.1032 | 0.000 | 0.000 | 1.000 | 0.000 | 0.0020 |
| Severity ridge | 0.0535 | 0.0167 | 1.000 | 1.000 | 1.000 | 5.350 | 0.0053 |
| Metadata ridge | 0.0190 | 0.0123 | 1.000 | 1.000 | 1.000 | 5.352 | 0.0059 |
| Image-stats ridge | 0.0110 | 0.0084 | 1.000 | 1.000 | 0.958 | 5.359 | 0.0053 |

The Exp06 row is included only as architectural context. It uses the same rebuilt scorer regime but does not have parameter-space MAE because it generates images directly rather than proxy parameters.

## Interpretation

Image-aware features reduce parameter error, which confirms that v2 successfully made the parameter target more image-dependent. However, this improvement does not translate into a decisive rendered-QA advantage because all learned controllers pass validation and test. The dominant limitation is now QA-gate saturation: the gate remains useful for rejecting failed approaches such as direct diffusion, but it is no longer discriminative enough to rank successful proxy controllers.

Metadata-ridge remains the most stable practical controller for the current renderer. It has slightly higher parameter error than image-stats ridge, but it preserves perfect monotonicity and reproduces the source-aware mask presets more cleanly. Image-stats ridge has the lowest parameter error but introduces one mild test monotonicity inversion.

The monotonicity inversion indicates that parameter-space accuracy and severity-order robustness are not identical objectives. Minimizing parameter MAE can recover the enriched target more closely while still producing a small ordering error under the scorer. This motivates a hybrid production architecture: image-aware parameter prediction followed by explicit monotonicity regularization or QA-constrained parameter refinement.

The inversion should be treated as a design signal rather than a trivial defect. The global image statistics may capture appearance cues that reduce parameter error while still creating a local scorer-ordering shortcut under cross-case generalization. Future feature engineering should therefore prioritize sparse, interpretable cues and include monotonicity-aware validation rather than optimizing parameter MAE alone.

## Image-Statistic Ablation

A follow-up ablation tested which of the nine image statistics actually drive the parameter-error improvement. The full nine-statistic model is not optimal. Most of the useful signal is concentrated in a sparse subset:

| Feature set | Val MAE | Test MAE | Test normalized MAE |
|---|---:|---:|---:|
| No image statistics | 0.0101 | 0.0229 | 0.0909 |
| All nine image statistics | 0.0110 | 0.0084 | 0.1240 |
| Hair-presence index only | 0.0035 | 0.0023 | 0.0423 |
| Hair-presence + masked gray mean | 0.0032 | 0.0022 | 0.0156 |
| Hair-presence + masked gray mean + mask area | 0.0032 | 0.0022 | 0.0154 |
| Hair-presence + masked gray mean + mask area + texture p90 | 0.0032 | 0.0022 | 0.0148 |

This ablation strengthens the image-aware claim while narrowing it. Image appearance matters, but a small interpretable feature subset is better suited to the current data scale than a larger correlated feature set. The best next controller should therefore use sparse image statistics plus explicit QA refinement, not a larger unconstrained feature vector by default.

The ablation also shows the current limit of the feature design. These are still global aggregate statistics; they do not encode local follicle distribution, density gradients, edge sharpness, or spatial thinning patterns. Richer spatial features may be useful later, but only if paired with monotonicity and QA constraints to avoid replacing pixel-space hallucination with parameter-space shortcut learning.

## Scientific Contribution

Experiment 07 changes the experimental trajectory. Experiments 03--06 showed that direct image generation is unreliable at the current data scale. Experiment 07 shows that learned control over a deterministic, QA-gated renderer is feasible and computationally successful.

The result supports a constrained-controller architecture for the thesis:

> learn parameter control, keep image formation deterministic and auditable, and use QA gates to reject invalid sweeps.

## Limitations

This remains proxy-supervised. The targets are not real clinical progression parameters and are not clinician-validated. The image-aware controls are deterministic functions of image statistics, so the experiment tests whether such controls can be learned, not whether they are clinically optimal. The learned controllers automate and improve control over the proxy mechanism, but they inherit the proxy renderer's clinical ceiling.

The evaluation set remains small. The finding should be interpreted as a design-space result: controller-based synthesis is more tractable than direct latent image generation under the current data regime.

The current QA gate may also be insufficiently discriminative for controller comparison. Severity-ridge, metadata-ridge, and image-stats ridge all achieve 100% auto-pass, which means the gate is useful for rejecting failed controllers but not strict enough to rank successful ones. Future work should add stricter secondary criteria such as LPIPS to proxy reference, hair-texture consistency, or exact monotonicity.

## Conclusion

Experiment 07 v2 confirms the promise of the learned proxy-controller path. Image-aware control improves parameter prediction, but metadata-aware control already achieves full image-level QA success. The learned controllers automate the proxy mechanism but do not remove its clinical limitations: the renderer itself still requires expert review or real progression supervision before any clinical-validity claim can be made. The next useful step is not another direct diffusion model; it is a QA-refined controller with explicit monotonicity and drift penalties, or a richer clinically motivated proxy parameter space.
