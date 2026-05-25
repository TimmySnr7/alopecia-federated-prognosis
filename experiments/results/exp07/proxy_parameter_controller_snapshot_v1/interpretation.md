# Experiment 07 Snapshot v1 Interpretation

## Verdict

Experiment 07 shows early promise as a pivot away from direct latent image generation and toward learned control of the validated proxy renderer.

The parameter-space controller succeeds on the first snapshot: a metadata-aware ridge model nearly exactly reconstructs the effective proxy-control vector on the fixed validation and test splits. This is a positive result, but it is a narrow one. It validates learnability of the current proxy parameterisation, not clinical image realism.

## Key Results

| Model | Val normalized MAE | Test normalized MAE | Test controller monotonicity | Test mask-area pass rate |
|---|---:|---:|---:|---:|
| Constant mean | 1.1099 | 0.5426 | 1.000 | 1.000 |
| Severity ridge | 0.7577 | 0.0850 | 1.000 | 1.000 |
| Metadata ridge | 0.00005 | 0.00004 | 1.000 | 1.000 |

The severity-only ridge model already captures nearly all target-dependent edit-strength controls. Its remaining error is mostly in the ellipse-mask coordinates, which depend on dataset/source mask presets rather than target severity. Adding dataset and schema metadata removes nearly all of this residual error.

## Scientific Meaning

This supports the central Experiment 07 hypothesis: the controllable part of the successful proxy mechanism lives in a low-dimensional space that can be learned with far less data than direct image generation requires.

Compared with Experiments 03--06, this formulation avoids hallucination by construction. The learned model does not synthesize pixels directly; it predicts transparent control parameters for a deterministic renderer.

## Main Limitation

The result is partially circular. The model is learning a parameter vector derived from the same deterministic proxy rules that produced the Experiment 02 targets. Therefore, this snapshot should be interpreted as a feasibility check, not as evidence of clinical generalization.

The local run also cannot evaluate rendered image quality because the manifest image paths point to AI5 paths under `/home/tmushuru/...`, which are not present locally.

## Next Step

Run Experiment 07 v2 on AI5:

1. Load held-out source images.
2. Predict proxy parameters using severity-only and metadata-aware controllers.
3. Render predicted proxy sweeps.
4. Evaluate using the full image-level QA gate: auto-pass, monotonicity, span, unmasked drift, mask area, and contact-sheet review.
5. Add an image-aware CNN+MLP controller only if metadata-aware rendering leaves measurable residual error.

The most important question for v2 is whether learned parameters preserve the Experiment 02 proxy pass rate when rendered, not whether the parameter MAE is low.
