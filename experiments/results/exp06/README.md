# Experiment 06 Results

Experiment 06 executed a stricter conditional diffusion baseline with explicit
severity-delta conditioning, ControlNet-style source-structure conditioning, and
an optional anatomical guard discriminator.

## Status

Closed as a documented negative result.

The model recovered more expected-severity span than Experiment 05, but it did
so by producing uncontrolled semantic content inside the scalp mask. Neither
variant matched the Experiment 02 proxy baseline or passed the generated-output
QA gate.

## Artifacts

- `plan.md`: pre-execution plan.
- `final_report.md`: milestone report and interpretation.
- `severity_delta_cross_attention_v1/summary.json`: primary severity-delta run.
- `severity_delta_cross_attention_v1/exp06_calibration.json`: run-start proxy
  calibration and determinism log.
- `severity_delta_controlnet_guard_v1/summary.json`: anatomical guard run.
- `severity_delta_controlnet_guard_v1/exp06_calibration.json`: run-start proxy
  calibration and determinism log.

Local contact sheets:

- `exp06_severity_delta_cross_attention_v1_samples/`
- `exp06_severity_delta_controlnet_guard_v1_samples/`

## Headline Test Results

| Variant | Auto-pass | Monotonic fraction | Expected-severity span | Proxy span | LPIPS to proxy | Proxy LPIPS | FID approx. | Unmasked drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Severity-delta cross-attention | 0.000 | 0.500 | 3.555 | 5.350 | 0.292 | 0.045 | 154.85 | 0.0318 |
| Severity-delta + guard | 0.000 | 0.417 | 3.395 | 5.350 | 0.305 | 0.045 | 169.24 | 0.0342 |

The proxy baseline auto-passed all validation/test cases under the rebuilt
Experiment 05 scorer. The Exp06 generated outputs auto-passed none.
