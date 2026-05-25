# Experiment 05 Plan

# Latent Conditional Diffusion Baseline

Date: 13 May 2026

## Status

Executed on 13 May 2026 and closed as a documented negative result. See
`final_report.md` for the run results and thesis-facing interpretation.

This is the first experiment that directly implements the core technical
artefact required by the approved PhD proposal: Contribution C1 and the Early
Experimentation Plan in Appendix A.

## Research Question

Can a custom latent conditional diffusion model, with multimodal conditioning
on the current scalp image, target severity, and optional clinical covariates,
generate calibrated, target-controlled severity sweeps that match or exceed the
Experiment 02 QA-gated proxy baseline in visual plausibility,
target-severity monotonicity, source identity preservation, and perceptual
quality?

This experiment directly delivers the severity-conditional synthesis baseline
defined in Appendix A.

## Why Experiment 05 Now

- Experiments 01 and 02 established a strong, auditable proxy baseline and
  QA-gated pseudo-pair supervision source.
- Experiment 03 showed that a small residual editor was visually plausible but
  under-expressive.
- Experiment 04 showed that a zero-shot pretrained latent editor was expressive
  but uncontrolled and prone to semantic hallucination.
- The project has therefore closed the proxy-engineering, residual-editor, and
  off-the-shelf-editor paths.
- The approved proposal and Appendix A require a latent conditional diffusion
  model with multimodal conditioning ablation as the first major contribution
  for C1 / RQ1.

## Data Scope

- Primary supervision: Experiment 02 QA-gated proxy-pair manifest, `78` rows
  from `13` source cases.
- Augmentation: heavy on-the-fly augmentation, including random crops, flips,
  colour jitter, and acquisition-style perturbations.
- Evaluation split: same validation/test source cases used in Experiments 03
  and 04, with `2` validation and `4` test cases.
- Scorer: rebuilt neural severity scorer, fine-tuned on proxy pairs and
  Experiment 03/04 outputs, used for monotonicity and calibration metrics.

## Method And Design

Primary backbone: Stable Diffusion v2.1 through Hugging Face Diffusers.

Adaptation method: LoRA / SeLoRA parameter-efficient fine-tuning. SDXL-Turbo is
reserved only as an optional secondary inference-speed comparison, not as the
primary backbone.

The planned implementation should bridge directly to proposal Contribution C1:
LoRA fine-tuning of a pretrained UNet with cross-attention severity injection,
plus image conditioning from the source scalp photograph. This is intended as
the first concrete instantiation of the proposal's latent conditional diffusion
architecture.

## Multimodal Conditioning Ablation

1. Cross-attention conditioning: severity token plus image features.
2. FiLM conditioning: feature-wise linear modulation on the UNet.
3. CLIP text conditioning: natural-language prompt encoding severity and
   covariates.

## Training Setup

- Fine-tune on the `78` proxy pairs, source image to proxy target.
- Use reconstruction and perceptual losses.
- Apply Experiment 02 soft ellipse masks during training and inference.
- Start with conservative LoRA rank and guidance settings to reduce
  instability.

## Run Variants

- Baseline latent diffusion adaptation without ablation.
- Cross-attention conditioning.
- FiLM conditioning.
- CLIP text conditioning.
- Conservative and stronger guidance settings.

## Evaluation Dimensions

Quantitative:

- LPIPS as the primary perceptual metric.
- Domain-adapted FID using a ResNet-18 fine-tuned on the full working scalp
  image pool of `57` images as the reference distribution, with the small
  reference set stated as a limitation.
- Expected-severity monotonic fraction and span using the rebuilt neural
  scorer.
- Unmasked/background drift, measured as mean absolute delta outside the mask.

Qualitative:

- Contact-sheet visual review.
- 2AFC clinician realism pilot with at least `3` dermatologists, using the
  separate 2AFC protocol.

Anatomical/localization guard:

- Add an edited-region hair/scalp plausibility check, using a region-level
  classifier or perceptual similarity to a scalp texture reference set.
- Add a contact-sheet review item that explicitly asks reviewers to flag
  non-scalp semantic content inside the mask.

Fairness:

- Report by Fitzpatrick skin tone where annotations exist.
- Fall back to source-dataset stratification where Fitzpatrick labels are
  absent or insufficient:
  `unidatapro_women_hair_loss`, `unidatapro_men_hair_loss`, and
  `unidpro_hair_loss_male_norwood_scale`.

Comparison:

- Report all metrics head-to-head against the Experiment 02 proxy baseline on
  the same validation/test cases.

## Success Criteria

The diffusion model is considered a successful baseline if it satisfies all of
the following:

1. Matches or exceeds the Experiment 02 proxy on mean expected-severity
   monotonic fraction and span using the rebuilt neural scorer.
2. Achieves LPIPS less than or equal to proxy baseline plus statistically
   comparable or better domain-adapted FID.
3. Maintains low unmasked drift: `<= 0.01` mean absolute delta.
4. Passes contact-sheet review with no systematic hallucinations or anatomical
   collapse.
5. Shows at least one conditioning strategy clearly superior in the ablation.

## Deliverables

- Trained latent diffusion model and LoRA weights.
- Full ablation results table and contact sheets.
- Updated neural severity scorer.
- 2AFC clinician realism pilot results with at least `3` dermatologists.
- Experiment 05 final report, including a threats-to-validity section.
- Updated GitHub milestone for Phase 2: Latent Conditional Diffusion Baseline.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Small data overfitting | LoRA + heavy augmentation + proxy-pair supervision |
| Training instability | Conservative guidance scale + smaller LoRA rank initially |
| Scorer quality | Neural scorer rebuild completed before evaluation phase |
| Clinician availability for 2AFC | Recruitment started this week; one-page 2AFC protocol prepared |

## Timeline

Weeks 1-3 from plan approval:

- Week 1: Set up Diffusers pipeline and complete neural scorer rebuild.
- Week 2: Run conditioning ablation and training.
- Week 3: Evaluation, 2AFC pilot, and report writing.

## Link To Approved Proposal

This experiment is explicitly required by Appendix A, the Early
Experimentation Plan, and directly implements the severity-conditional
synthesis baseline and multimodal conditioning ablation described in Section
3.5, RQ1, and C1 of the approved PhD proposal.
