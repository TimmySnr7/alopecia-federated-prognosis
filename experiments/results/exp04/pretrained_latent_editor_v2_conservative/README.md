# Pretrained Latent Editor (v2 Conservative)

## Purpose

This run tested whether the v1 failure was mainly caused by excessive edit
strength. It used the same pretrained `timbrooks/instruct-pix2pix` backbone but
increased image preservation and reduced the mask blend.

## Configuration

- Backbone: `timbrooks/instruct-pix2pix`
- Image size: `256`
- Inference steps: `20`
- Text guidance scale: `5.0`
- Image guidance scale: `2.5`
- Mask blend strength: `0.35`
- Identity target handling: source passthrough
- Scorer checkpoint:
  `/home/tmushuru/datasets/alopecia_public/metadata/exp01_working_severity_baseline_v4.ckpt`

## Results

| Split | Latent pass rate | Proxy pass rate | Latent mean monotonicity | Proxy mean monotonicity | Latent mean span | Proxy mean span | Mean latent max unmasked delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | `0.000` | `1.000` | `0.500` | `0.917` | `0.357` | `1.816` | `0.00156` |
| Test | `0.000` | `1.000` | `0.458` | `1.000` | `0.356` | `2.276` | `0.00650` |

## Interpretation

The conservative setting reduced unmasked drift and softened the worst
hallucinations, but it returned to the Experiment 03-style failure mode:
target-severity span collapsed and no validation or test case passed the
automatic gate.

The remaining contact-sheet artefacts show that even conservative prompting can
introduce face-like structures into the scalp region. The model is therefore
not acceptable as a direct zero-shot clinical editor, and the failure is not
solved by simple guidance-scale tuning.

## Contact Sheets

- `/home/tmushuru/datasets/alopecia_public/metadata/exp04_pretrained_latent_editor_v2_conservative_samples/val_latent_vs_proxy_contact_sheet.png`
- `/home/tmushuru/datasets/alopecia_public/metadata/exp04_pretrained_latent_editor_v2_conservative_samples/test_latent_vs_proxy_contact_sheet.png`

Local copies may exist under
`exp04_pretrained_latent_editor_v2_conservative_samples/`.
