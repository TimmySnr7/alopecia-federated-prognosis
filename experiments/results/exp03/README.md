# Experiment 03 Results

Experiment 03 tests whether a learned residual editor can inherit the
Experiment 02 proxy's target-controlled, mask-localized visual edits from the
QA-gated proxy-pair manifest.

Result packages:

- `proxy_residual_editor_v1/`: first learned residual-editor run trained on the
  `78` QA-gated proxy-pair rows.
- `proxy_residual_editor_v2/`: under-conditioning countermeasure run with
  severity-delta weighting, dataset-balanced sampling, stronger residual
  reconstruction, and scorer-space consistency.

Planned v2 run:

```bash
python experiments/scripts/train_exp03_proxy_residual_editor.py \
  experiments/results/exp02/source_aware_proxy_batch_v1/proxy_pair_manifest.csv \
  --scorer-checkpoint /home/tmushuru/datasets/alopecia_public/metadata/exp01_working_severity_baseline_v4.ckpt \
  --output-json experiments/results/exp03/proxy_residual_editor_v2/summary.json \
  --checkpoint-path /home/tmushuru/datasets/alopecia_public/metadata/exp03_proxy_residual_editor_v2.ckpt \
  --sample-output-dir /home/tmushuru/datasets/alopecia_public/metadata/exp03_proxy_residual_editor_v2_samples \
  --include-texture-channels \
  --epochs 75 \
  --batch-size 8 \
  --residual-scale 0.5 \
  --proxy-loss-weight 1.0 \
  --masked-proxy-loss-weight 1.0 \
  --delta-loss-weight 1.0 \
  --unmasked-loss-weight 5.0 \
  --target-delta-weight-strength 3.0 \
  --sampling-strategy dataset \
  --scorer-consistency-weight 0.05 \
  --expected-consistency-weight 0.1
```

The v2 hypothesis is that the v1 editor collapsed because low average
reconstruction error could be achieved with source-preserving muted residuals.
This run explicitly weights larger source-to-target severity shifts, balances
the imbalanced source datasets during sampling, and adds weak scorer-space
matching to recover target-severity separation while keeping the proxy L1 and
unmasked-drift safeguards in place.

Final Experiment 03 decision:

Experiment 03 is closed as a negative learned-editor result. Both v1 and v2
preserved visual plausibility better than earlier learned scaffolds, but neither
matched the Experiment 02 proxy baseline's target-severity separation. V2 tested
the most direct under-conditioning fixes and still produced `0.000` learned
auto-pass on validation and test, with test mean expected span collapsing to
`0.043` versus the proxy baseline's `2.276`.

The next experiment should prioritize the pretrained latent conditional
diffusion path from the proposal's Contribution C1, using mask-localized
adaptation and the Experiment 02 QA-gated proxy pairs as weak supervision,
rather than further tuning this residual-editor formulation.

Large visual contact sheets and model checkpoints are not tracked in Git. The
machine-readable summaries and interpretation notes are tracked here.
