# ChatGPT Handoff — Experiment 07

## Project

Alopecia Federated Prognosis

## Current Experimental State

Experiments 01--02 established a QA-gated deterministic proxy baseline for alopecia severity-sweep synthesis.

Experiments 03--06 tested increasingly constrained learned generative approaches:

- Exp03: residual editor, visually plausible but under-expressive.
- Exp04: zero-shot latent editor, expressive but hallucinatory.
- Exp05: LoRA latent adaptation, severity/plausibility trade-off but no full QA pass.
- Exp06: severity-delta conditional diffusion with weak guard, stronger scorer response but non-scalp hallucination and failed source preservation.

The conclusion from Exp03--06 is that direct learned image generation is not currently viable at the available data scale. The validated proxy renderer remains the strongest controllable mechanism.

## Experiment 07 Goal

Test whether a learned model can control the successful proxy mechanism by predicting low-dimensional proxy parameters instead of generating final images directly.

Research question:

> Can a learned controller operating in proxy-parameter space produce severity sweeps that preserve the QA validity of the Experiment 02 proxy baseline while reducing dependence on hand-specified parameters?

## Snapshot Already Run

Script:

`experiments/scripts/train_exp07_proxy_parameter_controller.py`

Output directory:

`experiments/results/exp07/proxy_parameter_controller_snapshot_v1/`

Input manifest:

`experiments/results/exp02/source_aware_proxy_batch_v1/proxy_pair_manifest.csv`

Snapshot target vector:

- `ellipse_cx`
- `ellipse_cy`
- `ellipse_rx`
- `ellipse_ry`
- `mask_blur_radius`
- `texture_blur_radius`
- `lower_hair_enhancement_eff`
- `lower_tone_shift_eff`
- `upper_hair_suppression_eff`
- `upper_smoothing_eff`
- `upper_tone_shift_eff`

The effective edit controls are severity-scaled versions of the Experiment 02 proxy parameters.

## Snapshot Results

The snapshot compared:

- constant mean baseline
- severity-only ridge controller
- metadata-aware ridge controller

Headline results:

- metadata-aware ridge validation normalized MAE: `5.109e-05`
- metadata-aware ridge test normalized MAE: `3.709e-05`
- metadata-aware ridge test controller monotonicity: `1.000`
- metadata-aware ridge test mask-area gate pass rate: `1.000`

Interpretation:

The proxy parameter space is learnable and low-dimensional. Most target-dependent controls are deterministic from source/target severity; dataset/schema metadata recovers mask-preset differences. This is promising, but partially circular because the target vector is derived from the deterministic proxy itself.

## Important Limitation

The local machine does not contain the AI5 image paths in the manifest:

`/home/tmushuru/datasets/alopecia_public/...`

Therefore snapshot v1 did not render predicted outputs or run image-level QA. Rendering must happen on AI5 or after syncing the image data locally.

## Recommended Next Step

Run Experiment 07 v2 on AI5:

1. Copy/pull the new script to AI5.
2. Run the parameter controller snapshot there to confirm identical parameter results.
3. Extend the script to render predicted proxy sweeps using the existing Exp01/Exp03 proxy utilities.
4. Evaluate rendered outputs with the same QA gate used in Experiment 02:
   - auto-pass rate
   - expected-severity monotonicity
   - expected-severity span
   - unmasked drift
   - mask-area fraction
   - contact-sheet review
5. Compare severity-only, metadata-aware, and eventually image-aware CNN+MLP controllers.

## Decision Rule

Proceed to image-aware CNN+MLP only if rendered metadata-aware predictions do not already match the proxy baseline on held-out source cases. If metadata-aware prediction renders perfectly, the next scientific question is not model capacity but whether the proxy parameter space should be made richer and less rule-determined.

## Files To Inspect First

- `experiments/scripts/train_exp07_proxy_parameter_controller.py`
- `experiments/results/exp07/proxy_parameter_controller_snapshot_v1/summary.json`
- `experiments/results/exp07/proxy_parameter_controller_snapshot_v1/interpretation.md`
- `experiments/results/exp02/source_aware_proxy_batch_v1/proxy_pair_manifest.csv`
- `experiments/scripts/run_exp01_plausible_proxy_sweep.py`
- `experiments/scripts/run_exp01_proxy_batch_eval.py`
- `experiments/scripts/train_exp03_proxy_residual_editor.py`
