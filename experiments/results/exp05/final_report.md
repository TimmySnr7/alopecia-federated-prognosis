# Experiment 05 Final Report

# Latent Conditional Diffusion Baseline

Date: 13 May 2026

## Status

Closed as a documented negative result.

Experiment 05 is the first executable attempt at the proposal's Contribution C1:
a severity-conditioned latent diffusion model trained from the Experiment 02
QA-gated proxy supervision set. The experiment completed the computational
pipeline for the planned baseline: scorer rebuild, LoRA latent-editor training,
conditioning ablation, proxy comparison, contact-sheet review, drift analysis,
perceptual metrics, and source-dataset stratification.

The experiment did not produce a successful baseline. No trained variant matched
or exceeded the Experiment 02 proxy baseline on the primary success criteria.

## Research Question

Can a domain-adapted latent conditional diffusion model, conditioned on the
current scalp image and target severity, generate calibrated severity sweeps
that match or exceed the Experiment 02 QA-gated proxy baseline in visual
plausibility, target-severity monotonicity, source preservation, and perceptual
quality?

Answer: no, not in this small-data LoRA configuration. The model can be trained
and evaluated end to end, but the learned latent editor either under-expresses
severity or introduces anatomically implausible semantic content inside the
scalp mask.

## Data And Split

Primary supervision came from the Experiment 02 QA-gated proxy-pair manifest:
`experiments/results/exp02/source_aware_proxy_batch_v1/proxy_pair_manifest.csv`.
The experiment used the same source-case split as Experiments 03 and 04:

- Training: `42` source-to-proxy pairs.
- Validation: `2` source cases.
- Test: `4` source cases.

The rebuilt scorer used `49` training examples, `14` validation examples, and
`28` test examples after adding source-identity anchors alongside proxy-pair
examples. The validation/test splits were source-case held out, preserving the
Experiment 03/04 evaluation protocol.

## Implementation

The planned backbone was Stable Diffusion v2.1 with image conditioning. On AI5,
an accessible SD2.1 image-conditioned checkpoint was not available, so the
executable baseline used `timbrooks/instruct-pix2pix` through Hugging Face
Diffusers as the pretrained image-conditioned latent editor. This keeps the
experiment within the latent-diffusion C1 path, but it is not a perfect
implementation of the planned SD2.1 backbone.

The following variants were trained with LoRA:

- Cross-attention-style severity conditioning.
- FiLM-style conditioning-text ablation.
- CLIP natural-language severity conditioning.

Each variant used LoRA rank `4`, alpha `4`, learning rate `0.00002`, two
epochs, batch size `1`, gradient accumulation `2`, and mask-aware training and
inference. Initial fp16 training produced instability, so final runs used fp32.

Important implementation limitation: the FiLM variant is an executable
conditioning ablation represented through prompt/conditioning text rather than a
custom FiLM-modified UNet. A true FiLM block remains a deeper architecture
change.

## Rebuilt Severity Scorer

The neural severity scorer was rebuilt before diffusion evaluation.

- Best validation macro-F1: `0.519`.
- Validation accuracy: `0.571`.
- Test macro-F1: `0.334`.
- Test accuracy: `0.393`.

This scorer is adequate for a consistent internal comparison across proxy and
latent outputs, but it remains a weak evaluation instrument. The test macro-F1
should be treated as a limitation when interpreting monotonicity and
expected-severity span.

Calibration note: Experiment 05 re-evaluates both the proxy baseline and the
latent outputs with this rebuilt neural scorer. The proxy spans reported here
(`4.458` validation and `5.350` test) are therefore on a different scale from
the Experiment 02-04 feature-scorer spans (`1.816` validation and `2.276` test).
The correct cross-experiment interpretation is that the proxy remains strongly
separated under both scorers, while the learned editors remain far below the
proxy on whichever scorer was used within a given experiment.

## Quantitative Results

| Variant* | Split | Auto-pass | Monotonic | Span | Proxy span | LPIPS to proxy | Proxy LPIPS | FID approx. | Unmasked drift | Guard L1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cross-attention | val | 0.000 | 0.333 | 0.209 | 4.458 | 0.050 | 0.027 | 15.16 | 0.0022 | 0.024 |
| Cross-attention | test | 0.000 | 0.542 | 1.156 | 5.350 | 0.108 | 0.045 | 40.90 | 0.0085 | 0.052 |
| FiLM-style | val | 0.000 | 0.583 | 0.152 | 4.458 | 0.039 | 0.027 | 13.76 | 0.0014 | 0.022 |
| FiLM-style | test | 0.000 | 0.625 | 0.585 | 5.350 | 0.080 | 0.045 | 34.73 | 0.0064 | 0.033 |
| CLIP text | val | 0.000 | 0.667 | 0.120 | 4.458 | 0.052 | 0.027 | 15.22 | 0.0040 | 0.033 |
| CLIP text | test | 0.000 | 0.458 | 1.150 | 5.350 | 0.104 | 0.045 | 36.78 | 0.0106 | 0.055 |

*FiLM-style is a prompt/conditioning-text proxy for FiLM, not a true
FiLM-modulated UNet. Its results should not be interpreted as evidence that a
proper FiLM architecture would behave the same way.

The Experiment 02 proxy baseline auto-passed all validation and test cases under
the same evaluation protocol. Every latent-diffusion variant failed the
auto-pass gate on both validation and test.

## Qualitative Review

Contact-sheet review shows a consistent trade-off:

- FiLM-style conditioning is the most conservative and has the lowest test
  LPIPS/FID approximation, but it mostly preserves the source appearance and
  does not recover proxy-level severity span.
- Cross-attention and CLIP conditioning recover more expected-severity span on
  test, but both introduce non-scalp semantic content in the edited region.
- The most common anatomical failure is face/bald-head hallucination inside the
  scalp mask, especially on female top-down source cases and larger target
  severity shifts.

In the cross-attention test contact sheet, `3/4` held-out test cases show
visibly implausible non-scalp content inside the edited region. Across the full
six-case validation/test evaluation set, this is `3/6` cases, because the two
validation cases did not show the same semantic hallucination pattern. The
hallucinations correlate with larger source-to-target shifts and are most
visible at higher target severities, although occasional lower-target artefacts
also occur.

The validation cases were two centred male Norwood top-down images with smaller
soft masks, while the test set included three women-hair-loss top-down images
with larger masked regions and stronger hair/background structure. The most
likely explanation for the validation/test asymmetry is that the two validation
cases were easier for mask compositing to contain and too small to represent the
harder acquisition conditions in the test set.

The FiLM-style run reduced but did not eliminate hallucination. It showed
visible face-like content in `2/4` test cases while also collapsing much of the
target severity range. The CLIP run showed the same mixed failure mode and added
an isolated colour artefact in one test case.

## Anatomical Guard

The planned anatomical/localization guard was operationalised in this run as:

- masked L1 distance to the QA-gated proxy target;
- contact-sheet review with explicit attention to non-scalp semantic content;
- unmasked drift against the Experiment 02 `<= 0.01` threshold.

This is a first operational guard, not a final clinical guard. A stronger
Experiment 06 guard should add a trained hair/scalp-region plausibility
classifier or texture-reference similarity model. The current metrics caught
the broad failure pattern, but visual review was still required to identify the
semantic nature of the hallucinations.

## Fairness / Stratification

Fitzpatrick annotations were not available at sufficient density for reliable
skin-tone stratification. The fallback source-dataset stratification was
therefore used.

Test split by source dataset:

- `unidatapro_men_hair_loss`: one held-out case.
- `unidatapro_women_hair_loss`: three held-out cases.

The women-hair-loss subset showed the strongest tension between recovered span
and hallucination. The single men-hair-loss case was more visually stable but
under-expressive, especially for FiLM-style and CLIP conditioning. Because the
test split contains only four source cases, this is descriptive rather than a
statistical fairness conclusion.

## 2AFC Clinician Pilot

The clinician 2AFC component could not be completed within the same-day
computational run because it requires external dermatologist responses. The
experiment produced the necessary contact-sheet artefacts for a 2AFC pilot, but
no clinician realism result is claimed here.

Given the visible hallucination rate in the contact sheets and the `0.000`
auto-pass rate across all variants, the computational milestone can be closed
without waiting for clinician votes. A clinician pilot remains valuable for
thesis-facing validation of the failure taxonomy, not for deciding whether this
baseline succeeded.

## Success Criteria Assessment

| Criterion | Result |
| --- | --- |
| Match/exceed proxy monotonicity and span | Failed. Best test monotonicity was `0.625`; proxy was `1.000`. Best test span was `1.156`; proxy was `5.350`. |
| LPIPS comparable to proxy | Failed. Every test variant had higher LPIPS than proxy. |
| Unmasked drift `<= 0.01` | Mostly passed. Cross-attention and FiLM-style stayed below threshold; CLIP slightly exceeded it at `0.0106`. |
| No systematic hallucination/anatomical collapse | Failed. Stronger variants produced face/bald-head hallucinations inside the mask. |
| One conditioning strategy clearly superior | Failed as a success criterion, but informative as a trade-off: FiLM-style dominated the plausibility dimensions, while cross-attention gave the strongest severity response; no single strategy dominated both plausibility and severity control. |

## Interpretation

Experiment 05 confirms that simply adding small-data LoRA adaptation to a
pretrained latent image editor is not sufficient for calibrated
severity-conditioned alopecia synthesis. The result is sharper than Experiment
04 because the model was no longer zero-shot: it was trained on the QA-gated
proxy pairs and still failed to dominate the proxy.

The central failure mode is not identical to Experiment 03 or Experiment 04.
Experiment 03 showed source-preserving under-expression. Experiment 04 showed
zero-shot expressive hallucination. Experiment 05 shows that lightweight
domain adaptation creates a continuum between those failures: conservative
conditioning remains under-expressive, while more expressive conditioning
revives semantic hallucination.

This is a scientifically useful negative result. It narrows the next step: the
project should not treat LoRA on a generic image editor as the final C1
architecture. The next experiment should use a more explicitly constrained
conditional diffusion formulation, with stronger anatomical conditioning and a
more reliable severity-evaluation stack.

## Threats To Validity

- The executable backbone was InstructPix2Pix rather than the planned SD2.1
  image-conditioned backbone.
- The FiLM condition is an ablation proxy, not a true FiLM-modulated UNet.
- The scorer remains weak on held-out test data, with macro-F1 `0.334`.
- The FID-like metric uses ResNet18 proxy-reference features rather than a
  fully fine-tuned 57-image scalp-domain FID model.
- The 2AFC clinician realism pilot is prepared but not completed.
- The validation/test set is small: `2` validation and `4` test source cases.
- The anatomical guard still depends partly on human contact-sheet review.

## Conclusion

Experiment 05 should be closed as a documented negative result. It satisfies the
need to execute a first latent conditional diffusion baseline for C1, but it
does not satisfy the success criteria for replacing or beating the Experiment
02 proxy baseline.

The result motivates a stricter next formulation: not another generic LoRA
tuning pass, but a conditional diffusion architecture with explicit severity
control, source preservation, anatomical/hair-region constraints, and a stronger
evaluation scorer before clinical 2AFC deployment.

The most specific Experiment 06 direction is a domain-specific conditional UNet,
initialised either from a medical-imaging foundation model or trained from
scratch on scalp imagery, with explicit severity-delta conditioning at the
feature level and a hair/scalp texture discriminator or region classifier to
penalise non-scalp content inside the mask.
