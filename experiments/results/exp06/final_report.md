# Experiment 06 Final Report

# Domain-Specific Conditional Diffusion Model With Explicit Anatomical Guard

Date: 13 May 2026

## Status

Closed as a documented negative result.

Experiment 06 executed the stricter architecture recommended after Experiment
05: explicit severity-delta conditioning, ControlNet-style source-structure
conditioning, source-aware sampling, proxy calibration before training, and an
optional edited-region anatomical guard discriminator.

The experiment did not produce a successful C1 baseline. Both variants failed
the generated-output QA gate on validation and test.

## Research Question

Can a domain-specific conditional diffusion model, with explicit severity-delta
conditioning and an anatomical guard, generate calibrated severity sweeps that
match or exceed the Experiment 02 QA-gated proxy baseline?

Answer: no, not in this implementation. Severity-delta conditioning increased
the scorer span relative to Experiment 05, but the model achieved that response
through uncontrolled semantic generation rather than clinically plausible
scalp/hair edits.

## Data And Calibration

Primary supervision came from the Experiment 02 QA-gated proxy-pair manifest:
`experiments/results/exp02/source_aware_proxy_batch_v1/proxy_pair_manifest.csv`.

The same source-case split as Experiments 03-05 was used:

- Training: `42` proxy pairs.
- Validation: `2` source cases.
- Test: `4` source cases.

Proxy targets were regenerated deterministically from manifest
`proxy_parameters`, `ellipse_mask`, and `image_path`. The run-start determinism
check confirmed sampled source/proxy/mask maximum absolute deltas of `0.0`.

Proxy calibration under the rebuilt Experiment 05 neural scorer:

- Validation proxy auto-pass: `1.000`; proxy span: `4.458`.
- Test proxy auto-pass: `1.000`; proxy span: `5.350`.

These spans are on the rebuilt neural-scorer scale, not the Experiment 02-04
feature-scorer scale.

## Implementation

The planned backbone was SD2.1 plus SD2.1 ControlNet. AI5 could not access
`stabilityai/stable-diffusion-2-1` or `stabilityai/stable-diffusion-2-1-base`
through Hugging Face without authentication. The executable run therefore used:

- Base model: `runwayml/stable-diffusion-v1-5`.
- ControlNet: `lllyasviel/control_v11p_sd15_softedge`.

This is a material limitation. It still tests the stricter conditional
diffusion formulation, but it is not the exact SD2.1 backbone specified in the
plan.

Both variants used:

- explicit severity-delta projection appended as a cross-attention token;
- ControlNet soft-edge source-structure hints;
- mask-aware reconstruction and source-preservation losses;
- LoRA rank `8`, alpha `8`;
- `8` epochs, batch size `1`, gradient accumulation `4`;
- generated validation/test sweeps after training.

Run variants:

- `severity_delta_cross_attention`: severity-delta token plus source-structure
  ControlNet.
- `severity_delta_controlnet_guard`: same architecture plus weakly supervised
  scalp-region discriminator guard.

For the guard variant, the discriminator was trained with QA-gated proxy target
regions as positive examples and the model's decoded generated regions as
negative examples. It therefore learned a proxy-vs-generated distinction, not an
independently labelled scalp-vs-non-scalp semantic boundary.

## Quantitative Results

| Variant | Split | Auto-pass | Monotonic | Span | Proxy span | LPIPS to proxy | Proxy LPIPS | FID approx. | Unmasked drift | Guard L1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Severity-delta cross-attention | val | 0.000 | 0.500 | 2.060 | 4.458 | 0.101 | 0.027 | 28.96 | 0.0096 | 0.100 |
| Severity-delta cross-attention | test | 0.000 | 0.500 | 3.555 | 5.350 | 0.292 | 0.045 | 154.85 | 0.0318 | 0.149 |
| Severity-delta + guard | val | 0.000 | 0.500 | 1.932 | 4.458 | 0.097 | 0.027 | 26.87 | 0.0090 | 0.106 |
| Severity-delta + guard | test | 0.000 | 0.417 | 3.395 | 5.350 | 0.305 | 0.045 | 169.24 | 0.0342 | 0.177 |

Neither variant passed the generated-output QA gate. The proxy baseline
auto-passed every validation and test case.

## Qualitative Review

The contact sheets show a stronger and clearer failure than Experiment 05.

The severity-delta cross-attention variant often produced large scorer changes,
but many images contained non-scalp semantic content inside the edited region:
faces, helmet-like structures, text-like marks, masks, stylised heads, and
high-frequency synthetic patterns. These artefacts are not subtle; they are
visible in all four held-out test cases.

The anatomical guard variant did not suppress the failure. It produced similar
or worse non-scalp semantic content, and its test metrics degraded relative to
the primary variant:

- Test monotonicity fell from `0.500` to `0.417`.
- Test span fell from `3.555` to `3.395`.
- Test LPIPS worsened from `0.292` to `0.305`.
- Test unmasked drift worsened from `0.0318` to `0.0342`.
- Guard L1 worsened from `0.149` to `0.177`.

This indicates that the weak discriminator did not learn a useful scalp-region
prior from the QA-gated proxy targets alone.

## Success Criteria Assessment

| Criterion | Result |
| --- | --- |
| Match/exceed proxy monotonicity and span | Failed. Best test monotonicity was `0.500` vs proxy `1.000`; best test span was `3.555` vs proxy `5.350`. |
| LPIPS comparable to proxy | Failed. Best test LPIPS was `0.292`; proxy LPIPS was `0.045`. |
| Unmasked drift `<= 0.01` | Failed on test. Best test unmasked drift was `0.0318`, about `3.2x` the threshold. |
| Anatomical guard pass rate `>= 90%` | Failed. Contact sheets showed non-scalp semantic content in all four test cases. |
| One strategy clearly superior | Failed as a success criterion. The primary variant was less bad than the guard variant on test, but neither approached proxy performance. |

## Interpretation

Experiment 06 is scientifically important because it rules out a stronger
formulation than Experiment 05. The failure is no longer merely a limitation of
prompt conditioning or LoRA on an image editor. Even with explicit
severity-delta conditioning and source-structure ControlNet hints, the model
still relies on the general-image prior and fills the scalp region with
semantically inappropriate content.

The experiment also separates severity response from clinical plausibility. The
model can move the scorer more than Experiment 05, but the movement is achieved
by generating non-scalp objects rather than calibrated alopecia progression.
This means expected-severity span alone is insufficient; anatomical guard
metrics and visual review are essential for C1.

The source-preservation loss was active throughout training, with weight `0.3`,
and was computed every step through the decoded denoising prediction. Its
failure to keep test unmasked drift below `0.01` is therefore best interpreted
as evidence that the SD1.5/ControlNet general-image prior and inference-time
generation path overrode the auxiliary preservation constraint at this data
scale, rather than as a missing loss implementation.

## Threats To Validity

- The executable backbone was SD1.5, not the planned SD2.1, because SD2.1 was
  inaccessible on AI5 without Hugging Face authentication.
- The anatomical guard discriminator was weakly supervised by proxy targets and
  did not have independent labels for non-scalp semantic content.
- The scorer remains weak and can be fooled by non-scalp semantic changes.
- The FID-like metric uses ImageNet ResNet18 features rather than a scalp-domain
  feature model. The reference set and feature extraction pipeline are the same
  proxy-reference ResNet18 procedure used in Experiment 05, so the values are
  broadly comparable as an internal diagnostic, but they should not be treated
  as a clinically validated domain-FID.
- The validation/test set remains small: `2` validation and `4` test source
  cases.
- The 2AFC clinician pilot was not run because the generated outputs failed
  computational and visual QA decisively.

## Conclusion

Experiment 06 should be closed as a documented negative result. The stricter
conditional diffusion formulation recovered more severity response than
Experiment 05, but failed because the generated changes were not anatomically
or clinically plausible.

The next step should not be another general Stable Diffusion/ControlNet tuning
run. The most direct Experiment 07 direction is a learned severity controller
that operates in the validated proxy parameter space, controlling the composite
overlay parameters that produced the Experiment 02 QA-gated targets rather than
generating the final image directly. This separates severity control, a
regression or classification problem in parameter space, from image synthesis,
which the proxy mechanism already solves, and removes the dependency on a
scalp-domain image prior that the available data cannot currently provide.
