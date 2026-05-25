# Experiment 04 Final Report

## Status

Experiment 04 is complete as a pretrained latent editing feasibility
experiment.

It is a negative result for direct zero-shot use of a general pretrained
image-editing backbone, but it is scientifically useful because it separates two
failure modes that Experiments 03 and 04 now expose clearly:

- the residual editor was visually plausible but under-expressive,
- the pretrained latent editor was expressive but not target-controlled or
  clinically localized.

## Research Question

Can a pretrained latent image-editing backbone, prompted by source and target
alopecia severity and constrained by the Experiment 02 mask, beat the QA-gated
proxy baseline on target-controlled severity sweeps?

The answer is no under the current zero-shot prompted configuration.

## Data Scope

Experiment 04 used the same Experiment 02 QA-gated proxy-pair manifest and the
same validation/test source cases used by Experiment 03:

- Validation cases: `2`
- Test cases: `4`
- Severity targets per case: `7`
- Proxy baseline: Experiment 02 QA-gated source-aware proxy edits
- Scorer: Experiment 02/03 severity QA scorer

The Experiment 04 proxy baseline is evaluated only on QA-passing Experiment 02
cases, so the proxy pass rate is expected to remain `1.000` on validation and
test.

## Method

The experiment used `timbrooks/instruct-pix2pix` through `diffusers` as the
pretrained latent image-editing backbone. For each source case and target
severity, the script generated a severity-targeted prompt, ran image-to-image
editing, and composited the edited output back into the source image using the
Experiment 02 soft ellipse mask.

Identity targets were passed through unchanged so the sweep retained a stable
source anchor.

Two settings were run:

1. `pretrained_latent_editor_v1`: stronger edit setting with full mask blend.
2. `pretrained_latent_editor_v2_conservative`: stronger image preservation and
   lower mask blend.

## Results

| Run | Split | Latent pass rate | Proxy pass rate | Latent mean monotonicity | Proxy mean monotonicity | Latent mean span | Proxy mean span | Mean latent max unmasked delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | Validation | `0.000` | `1.000` | `0.500` | `0.917` | `1.226` | `1.816` | `0.00912` |
| v1 | Test | `0.000` | `1.000` | `0.417` | `1.000` | `1.212` | `2.276` | `0.03795` |
| v2 conservative | Validation | `0.000` | `1.000` | `0.500` | `0.917` | `0.357` | `1.816` | `0.00156` |
| v2 conservative | Test | `0.000` | `1.000` | `0.458` | `1.000` | `0.356` | `2.276` | `0.00650` |

## Interpretation

The pretrained latent editor did not beat the proxy baseline.

The v1 setting showed that the pretrained backbone can create large
scorer-visible severity shifts. This directly addresses the Experiment 03
under-expression problem, but it introduces a worse clinical-editing problem:
the generated content is not controlled by target severity and is not reliably
localized to plausible scalp changes. Contact sheets show face and bald-head
hallucinations inside the scalp mask, and test unmasked drift exceeds the QA
threshold.

Manual contact-sheet review found anatomically implausible face, eye, mouth, or
bald-head content in all `6 / 6` validation/test source cases in v1, mainly on
non-identity target shifts; the frequency increased with stronger edit
activation but did not follow a clean source-to-target severity-distance
pattern. The v1 non-monotonic transitions are therefore best interpreted as
erratic prompt-conditioned semantic jumps and high-severity saturation, not as
a systematic ordinal inversion, implying that the zero-shot conditioning signal
is not aligned with the alopecia severity axis.

On test, v1's mean max unmasked delta was `0.03795`, about `3.8x` the
Experiment 02 QA threshold of `<= 0.01`; v2 conservative reduced this to
`0.00650`, inside the threshold, but did so by collapsing mean expected-severity
span from `1.212` to `0.356`.

The conservative v2 setting reduced background drift and softened the most
extreme hallucinations, but the target-severity span collapsed to approximately
Experiment 03 levels. This shows the practical tradeoff: preserving source
plausibility removes most of the pretrained editor's severity response, while
allowing stronger edits breaks visual and anatomical plausibility.

## Main Findings

1. A general pretrained latent image editor is more expressive than the
   Experiment 03 residual editor.
2. Zero-shot prompt conditioning is not enough to produce ordered alopecia
   severity control.
3. Mask compositing alone does not prevent anatomically implausible latent
   hallucinations inside the allowed scalp region.
4. Conservative guidance settings reduce artefacts but return to severity-span
   collapse.
5. The Experiment 02 proxy baseline remains stronger for auditable
   target-controlled severity sweeps.

## What Experiment 04 Supports

Experiment 04 supports the broader latent-diffusion pivot, but not as a
zero-shot off-the-shelf editor.

The result suggests that a pretrained backbone may still be useful if adapted
with domain-specific controls: mask-aware fine-tuning, ControlNet-style spatial
conditioning, LoRA or similar lightweight adaptation on the QA-gated pairs, and
explicit losses that discourage non-scalp semantic hallucinations. In proposal
terms, the natural next step is LoRA fine-tuning of a pretrained UNet with
cross-attention severity injection, so Experiment 05 becomes the first direct
instantiation of Contribution C1 rather than a detour from it.

## What Experiment 04 Does Not Support

Experiment 04 does not support claiming:

- that a general pretrained image editor can be used directly for alopecia
  prognosis visualisation,
- that prompt wording plus mask compositing is sufficient clinical control,
- or that the pretrained latent approach already beats the Experiment 02 proxy
  baseline.

## Recommended Next Experiment

The next experiment should keep the pretrained latent backbone direction, but
move from zero-shot prompting to domain adaptation.

The highest-priority next design is a mask-aware lightweight adaptation
experiment, such as LoRA or ControlNet-style conditioning, trained on the
Experiment 02 QA-gated proxy pairs and evaluated with the same scorer-gated
sweeps. The key additional requirement is an anatomical/localization guard:
generated changes must stay hair/scalp-like inside the mask, not merely inside
the mask. Experiment 05 should operationalise this guard with an edited-region
hair/scalp plausibility check, combining a region-level classifier or reference
texture-similarity score with a contact-sheet review question that explicitly
flags non-scalp content inside the mask.

## Final Conclusion

Experiment 04 is a useful negative result.

It shows that pretrained latent editing solves the residual editor's lack of
expressiveness but introduces uncontrolled semantic hallucination and still
fails target-ordered severity control. The thesis trajectory remains coherent:
Experiment 03 closed the small residual-editor path, and Experiment 04 shows
that the latent diffusion path requires domain-specific adaptation rather than
zero-shot prompting.
