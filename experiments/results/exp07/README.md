# Experiment 07 — Learned Proxy-Parameter Controller

Experiment 07 is the pivot from direct learned image generation to learned control of the validated Experiment 02 proxy renderer.

The motivating finding from Experiments 03--06 is that learned image generators either preserve source appearance while under-expressing severity or increase severity-scorer response through anatomically invalid hallucination. Experiment 07 therefore moves the learned component into the low-dimensional proxy parameter space.

## Snapshot v1

`proxy_parameter_controller_snapshot_v1/` contains the first local feasibility run.

It evaluates whether severity-only and metadata-aware ridge controllers can predict the effective proxy-control vector from the Experiment 02 proxy-pair manifest.

The snapshot is promising but limited:

- the metadata-aware controller nearly exactly reconstructs the effective parameter vector;
- the result is partly expected because the current proxy is strongly deterministic;
- no rendered image-level QA was run locally because the source image paths exist on AI5, not on this machine.

The next step is an AI5 rendering run that evaluates whether predicted parameters preserve the full proxy QA pass rate on held-out source cases.
