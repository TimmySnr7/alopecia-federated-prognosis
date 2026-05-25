# Experiment 05 Results

Experiment 05 executed the first domain-adapted latent conditional diffusion
baseline for proposal Contribution C1. The experiment rebuilt the neural
severity scorer, trained LoRA latent-editor variants on the Experiment 02
QA-gated proxy pairs, evaluated the same validation/test source cases used in
Experiments 03 and 04, and produced quantitative summaries plus contact sheets.

## Status

Milestone status: closed as a documented negative result.

The strongest setting did not match or exceed the Experiment 02 proxy baseline.
All three conditioning variants had `0.000` latent auto-pass on validation and
test. The main failure mode was a mixed regime: weaker settings preserved
background and scalp structure but under-expressed severity, while stronger
settings recovered more scorer span but reintroduced semantic hallucinations
inside the scalp mask.

## Artifacts

- `plan.md`: pre-execution Experiment 05 plan.
- `final_report.md`: milestone report and thesis-facing interpretation.
- `proxy_severity_scorer_v1/summary.json`: rebuilt neural scorer summary.
- `lora_cross_attention_v1/summary.json`: cross-attention-style LoRA run.
- `lora_film_v1/summary.json`: FiLM-style conditioning-text ablation run.
- `lora_clip_v1/summary.json`: CLIP text conditioning run.

Large model weights and generated sample sheets were kept out of Git. The LoRA
weights were written on AI5 under:

- `/home/tmushuru/datasets/alopecia_public/metadata/exp05_lora_cross_attention_v1`
- `/home/tmushuru/datasets/alopecia_public/metadata/exp05_lora_film_v1`
- `/home/tmushuru/datasets/alopecia_public/metadata/exp05_lora_clip_v1`

Local pulled contact sheets:

- `exp05_lora_cross_attention_v1_samples/`
- `exp05_lora_film_v1_samples/`
- `exp05_lora_clip_v1_samples/`

## Headline Test Results

| Variant* | Auto-pass | Monotonic fraction | Expected-severity span | Proxy span | LPIPS to proxy | Proxy LPIPS | FID approx. | Unmasked drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cross-attention | 0.000 | 0.542 | 1.156 | 5.350 | 0.108 | 0.045 | 40.90 | 0.0085 |
| FiLM-style | 0.000 | 0.625 | 0.585 | 5.350 | 0.080 | 0.045 | 34.73 | 0.0064 |
| CLIP text | 0.000 | 0.458 | 1.150 | 5.350 | 0.104 | 0.045 | 36.78 | 0.0106 |

*FiLM-style is a prompt/conditioning-text proxy, not a true FiLM-modulated
UNet.

The proxy span values are on the rebuilt Experiment 05 neural-scorer scale, not
the Experiment 02-04 feature-scorer scale.

None of the variants met the success criteria. FiLM-style conditioning was the
least visually disruptive but still far below the proxy span. Cross-attention
and CLIP recovered more span, but at the cost of poorer perceptual fidelity and
visible non-scalp semantic content in contact-sheet review.
