# Pretrained Latent Editor (v1)

## Purpose

This run tested a pretrained latent image-editing backbone as the first
Experiment 04 pivot away from the Experiment 03 residual editor.

The implementation used `timbrooks/instruct-pix2pix`, severity-target text
instructions, and soft mask-localized compositing into the original source
image. It was evaluated on the same Experiment 02 QA-gated validation and test
cases used by Experiment 03.

## Configuration

- Backbone: `timbrooks/instruct-pix2pix`
- Image size: `256`
- Inference steps: `20`
- Text guidance scale: `7.5`
- Image guidance scale: `1.5`
- Mask blend strength: `1.0`
- Identity target handling: source passthrough
- Scorer checkpoint:
  `/home/tmushuru/datasets/alopecia_public/metadata/exp01_working_severity_baseline_v4.ckpt`

## Results

| Split | Latent pass rate | Proxy pass rate | Latent mean monotonicity | Proxy mean monotonicity | Latent mean span | Proxy mean span | Mean latent max unmasked delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | `0.000` | `1.000` | `0.500` | `0.917` | `1.226` | `1.816` | `0.00912` |
| Test | `0.000` | `1.000` | `0.417` | `1.000` | `1.212` | `2.276` | `0.03795` |

## Interpretation

The pretrained latent editor made large scorer-visible changes, unlike the
Experiment 03 residual editor. However, those changes were not controlled by
target severity. Several outputs jumped toward scorer severity `7` regardless
of target, and the test unmasked drift exceeded the `0.01` QA threshold.

Visual contact sheets show severe out-of-domain hallucinations: face and bald
head structures are generated inside the scalp mask. This is a stronger and
more expressive model than the residual editor, but not a clinically suitable
severity editor under this zero-shot prompted configuration.

## Contact Sheets

- `/home/tmushuru/datasets/alopecia_public/metadata/exp04_pretrained_latent_editor_v1_samples/val_latent_vs_proxy_contact_sheet.png`
- `/home/tmushuru/datasets/alopecia_public/metadata/exp04_pretrained_latent_editor_v1_samples/test_latent_vs_proxy_contact_sheet.png`

Local copies may exist under `exp04_pretrained_latent_editor_v1_samples/`.
