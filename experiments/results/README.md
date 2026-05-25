# Experiment Results

Current sequence:

- `exp01/`: reproducible pipeline and hand-crafted proxy baseline.
- `exp02/`: scorer-gated, source-aware, QA-gated proxy supervision.
- `exp03/`: residual editor negative result, visually plausible but
  under-expressive.
- `exp04/`: zero-shot pretrained latent editor negative result, expressive but
  uncontrolled.
- `exp05/`: first domain-adapted latent conditional diffusion baseline for
  proposal Contribution C1, closed as a negative result: LoRA adaptation did
  not beat the Experiment 02 proxy baseline and exposed an
  under-expression/hallucination trade-off.
- `exp06/`: stricter conditional diffusion model with explicit severity-delta
  conditioning and anatomical guard, closed as a negative result: it recovered
  more severity span but produced uncontrolled non-scalp semantic content.

Track large artefacts here with DVC or Git LFS. Do not commit model weights
directly to Git.

- `exp01/`: feasibility and first visual-plausibility baseline.
- `exp02/`: scorer-gated, source-aware, QA-gated proxy-supervision pipeline.
- `exp03/`: learned residual-editor experiments trained from the Experiment 02
  QA-gated proxy-pair manifest.
