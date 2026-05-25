# Experiment 09 v1: Fragility-Aware Evaluation

This run evaluates pipeline fragility under mask, target, image-acquisition, and proxy-parameter perturbations.

## Key Files

- `summary.json`: full machine-readable results.
- `fragility_by_pipeline_and_perturbation.csv`: all pipeline/split/perturbation scores.
- `mean_fragility_by_pipeline_family.csv`: grouped fragility by perturbation family.
- `component_breakdown.csv`: ordering, QA, and critical-error components.
- `weight_sensitivity.csv`: balanced, safety-heavy, QA-heavy, and ordering-heavy weighting.
- `human_face_validity_rating_template.csv`: ready-to-use rater sheet.
- `human_rating_correlation_table.csv`: placeholder for correlation after ratings are collected.
- `figure1_fragility_curves.svg`: fragility curves by perturbation strength.
- `figure2_stacked_component_bars.svg`: stacked component breakdown.

## Headline

The sparse controllers remain substantially less fragile than the archived diffusion editor. The lowest mean balanced fragility across all tested conditions is `P3_sparse2_ridge` at 0.0118, followed by `P4_sparse4_ridge` at 0.0157, the deterministic proxy at 0.0319, and the archived diffusion comparator at 0.5509.

The main sparse-controller weakness is proxy-parameter noise. Mask shifts, target noise, erosion/dilation, and nominal conditions are stable; brightness, contrast, and image noise introduce small fragility increases.
