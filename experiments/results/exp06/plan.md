# Experiment 06 Plan

# Domain-Specific Conditional Diffusion Model With Explicit Anatomical Guard

Date: 13 May 2026

## Status

Pre-execution, immediate next experiment.

Experiment 06 is the first strict C1 architecture test after Experiment 05
showed that lightweight LoRA adaptation of a general pretrained image editor is
not sufficient.

## Research Question

Can a domain-specific latent conditional diffusion model, initialised or
fine-tuned on scalp imagery, with explicit severity-delta conditioning and an
anatomical/hair-region guard, generate calibrated severity sweeps that match or
exceed the Experiment 02 QA-gated proxy baseline in visual plausibility,
target-severity monotonicity, source preservation, and perceptual quality?

## Why Experiment 06 Now

- Experiments 01 and 02 established the QA-gated proxy baseline and pseudo-pair
  supervision source.
- Experiment 03 showed that a residual editor can preserve plausibility but
  under-expresses severity.
- Experiment 04 showed that zero-shot latent editing is expressive but
  hallucinatory.
- Experiment 05 showed that LoRA adaptation of a general latent editor still
  fails to jointly control severity and preserve scalp plausibility.

The next experiment must therefore test a stricter, domain-specific conditional
diffusion formulation rather than another generic image-editor adaptation.

## Data Scope

- Primary supervision: Experiment 02 QA-gated proxy-pair manifest, `78` rows
  from `13` source cases.
- Additional supervision: Experiment 03-05 outputs may be used as negative
  examples for anatomical guard calibration, but not as positive targets.
- Evaluation split: same source-case validation/test split as Experiments 03-05:
  `2` validation cases and `4` test cases.
- Sampling: source-aware and dataset-balanced sampling to reduce dominance by
  any one public source subset.

## Method

Primary architecture:

- Stable Diffusion v2.1 latent UNet.
- ControlNet-style spatial conditioning from a source-preserving scalp
  structure hint.
- Explicit severity-delta projection appended to the text/cross-attention
  hidden states.
- LoRA on the UNet and optionally ControlNet; full fine-tuning is enabled only
  if compute is stable.
- Mask-aware reconstruction/source-preservation losses.
- Anatomical guard loss using a small edited-region discriminator or classifier
  to penalise non-scalp semantic content inside the mask.

Focused ablation:

1. `severity_delta_cross_attention`: severity-delta token plus source spatial
   structure.
2. `severity_delta_controlnet_guard`: severity-delta token plus source spatial
   structure and adversarial/perceptual anatomical guard.

## Evaluation

Use the Experiment 05 rebuilt neural scorer for internal consistency, with the
same calibration warning: these spans are on the rebuilt scorer scale, not the
Experiment 02-04 feature-scorer scale.

Quantitative:

- Auto-pass rate.
- Expected-severity monotonic fraction and span.
- LPIPS against the proxy target.
- ResNet/FID-style proxy-reference distance.
- Mean unmasked drift, threshold `<= 0.01`.
- Anatomical guard pass rate and masked plausibility distance.
- Dataset-stratified reporting where Fitzpatrick labels are unavailable.

Qualitative:

- Contact-sheet review with explicit non-scalp-content flagging.
- 2AFC clinician realism pilot, mandatory before thesis use.

## Success Criteria

The model is a successful C1 baseline only if all of the following hold:

1. Matches or exceeds the Experiment 02 proxy on monotonic fraction and span
   under the rebuilt neural scorer.
2. Achieves LPIPS less than or equal to the proxy baseline and comparable or
   better domain-FID approximation.
3. Maintains unmasked drift `<= 0.01`.
4. Achieves anatomical guard pass rate `>= 90%` on contact-sheet review.
5. Shows one conditioning/guard strategy that clearly dominates the alternatives
   across both severity response and plausibility.

## Deliverables

- Trained domain-specific conditional diffusion weights.
- Full ablation and anatomical guard results table.
- Contact sheets and 2AFC pilot package.
- Experiment 06 final report with threats-to-validity and C1 linkage.

## Pre-Run Checklist

Before launching the first full AI5 training run:

- Run an import smoke test for Diffusers ControlNet, PEFT, PIL, torchvision, and
  the project proxy utilities.
- Confirm `torch.cuda.is_bf16_supported()` on AI5 and fall back to fp32 if
  needed.
- Verify proxy determinism from the manifest-provided `proxy_parameters`; the
  training script now checks sampled rows and aborts if regenerated proxy
  targets differ.
- Write calibration before training to
  `checkpoint_dir/exp06_calibration.json`, including proxy validation/test span,
  pass rate, scorer checkpoint, and determinism check.
- Save LoRA weights, severity projection weights, optional ControlNet weights,
  and optional discriminator weights as separate artefacts.
- Ensure ControlNet hints are always 3-channel.
- Keep discriminator and UNet/LoRA optimisation isolated with detached fake
  images for the discriminator update and frozen discriminator weights during
  generator guard loss.

## Link To Approved Proposal

Experiment 06 is the direct continuation of Section 3.5, RQ1, C1, and Appendix
A. It operationalises the empirically justified reference architecture after the
project ruled out proxy-only, residual-editor, zero-shot latent-editor, and
lightweight-LoRA formulations.
