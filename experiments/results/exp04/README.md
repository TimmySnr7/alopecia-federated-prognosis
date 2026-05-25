# Experiment 04 Results

Experiment 04 tests the pivot recommended after Experiment 03: a pretrained
latent image-editing backbone, conditionally prompted by source and target
severity, evaluated against the Experiment 02 QA-gated proxy baseline.

Result packages:

- `pretrained_latent_editor_v1/`: InstructPix2Pix baseline with full
  mask-localized compositing.
- `pretrained_latent_editor_v2_conservative/`: conservative ablation with
  stronger image preservation and lower mask blend.

Large visual contact sheets and downloaded model weights are not tracked in
Git. Machine-readable summaries and interpretation notes are stored in each run
package.

Final Experiment 04 decision:

Experiment 04 is closed as a negative but highly informative pretrained latent
editing result. The pretrained backbone can make large scorer-visible changes,
unlike the Experiment 03 residual editor, but those changes are not reliably
target-conditioned or clinically localized. The aggressive setting produced
severe hallucinated bald-head/face artefacts inside the scalp mask, while the
conservative setting reduced artefacts at the cost of returning to
target-severity collapse.
