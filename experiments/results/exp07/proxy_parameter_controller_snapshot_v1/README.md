# Experiment 07 Snapshot v1 — Proxy-Parameter Controller

This snapshot tests whether the Experiment 02 QA-gated proxy parameter space is learnable before moving to image-aware rendering.

The script predicts an effective proxy-control vector from the proxy-pair manifest:

- ellipse mask coordinates
- blur constants
- severity-scaled hair enhancement, suppression, smoothing, and tone controls

Because the local machine does not contain the AI5 image paths referenced by the manifest, this snapshot evaluates severity-only and metadata-aware parameter controllers only. It does not render predicted outputs or run image-level QA.

## Command

```bash
/Users/timothymusharu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  experiments/scripts/train_exp07_proxy_parameter_controller.py
```

## Outputs

- `summary.json` — split summaries, parameter errors, and controller diagnostics.
- `parameter_predictions.csv` — true and predicted effective parameter vectors.

## Headline Result

The metadata-aware ridge controller almost exactly recovers the effective proxy parameter vector on validation and test:

- validation normalized MAE: `5.109e-05`
- test normalized MAE: `3.709e-05`
- test predicted-effect monotonicity: `1.000`
- test mask-area gate pass rate: `1.000`

The severity-only controller also recovers the severity-scaled edit controls very well, but it cannot fully recover split-specific ellipse-mask preset differences.

## Interpretation

The snapshot supports the Experiment 07 pivot: the proxy parameter space is low-dimensional and learnable. However, this v1 result should not be overstated. The current proxy parameters are strongly deterministic, so success here mostly shows that the controller can reproduce the known proxy rule and dataset/mask preset structure.

The next step is an AI5 run that renders predicted parameters on held-out source images and evaluates the resulting sweeps using the full image-level QA gate.
