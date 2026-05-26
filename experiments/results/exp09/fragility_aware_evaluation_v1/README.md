# Experiment 09 v1: Fragility-Aware Evaluation

This run evaluates pipeline fragility under mask, target, image-acquisition, and proxy-parameter perturbations.

## Key Files

- `summary.json`: full machine-readable results.
- `fragility_by_pipeline_and_perturbation.csv`: all pipeline/split/perturbation scores.
- `mean_fragility_by_pipeline_family.csv`: grouped fragility by perturbation family.
- `component_breakdown.csv`: ordering, QA, and critical-error components.
- `weight_sensitivity.csv`: balanced, safety-heavy, QA-heavy, and ordering-heavy weighting.
- `protocol_sensitivity_sparse_vs_proxy.csv`: test-only protocol-sensitivity summary for the proxy-versus-sparse comparison.
- `human_face_validity_rating_template.csv`: ready-to-use rater sheet.
- `human_rating_correlation_table.csv`: placeholder for correlation after ratings are collected.
- `figure1_fragility_curves.svg`: fragility curves by perturbation strength.
- `figure2_stacked_component_bars.svg`: stacked component breakdown.

## Headline

Among the matched perturbation-tested pipelines, the lowest mean balanced fragility across all tested conditions is `P3_sparse2_ridge` at 0.0118, followed by `P4_sparse4_ridge` at 0.0157 and the deterministic proxy at 0.0319. The archived diffusion comparator is reported at 0.5509 as a high-fragility reference from Exp06, not as a fresh per-perturbation rerun.

The main sparse-controller weakness is proxy-parameter noise. Mask shifts, target noise, erosion/dilation, and nominal conditions are stable; brightness, contrast, and image noise introduce small fragility increases.

The sparse-versus-proxy comparison is protocol-sensitive because perturbations do not enter the deterministic proxy and ridge-controller pathways in the same way. The added protocol-sensitivity CSV should be used when discussing whether sparse2 is genuinely more robust than the proxy or simply benefits from regularised parameter prediction under this perturbation design.
