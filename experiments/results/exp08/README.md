# Experiment 08 Results

Experiment 08 evaluates whether a QA-constrained hybrid controller can resolve the Experiment 07 v2 trade-off between parameter-space accuracy and severity-order robustness.

The executed run is:

- `qa_refined_proxy_controller_v1`: sparse image-stat ridge controllers plus bounded QA refinement over the sparse-4 controller.

Primary conclusion: the sparse image-stat controllers already restore perfect held-out monotonicity while retaining low proxy error. The post-prediction refinement loop is feasible and preserves QA validity, but it is not needed under the current renderer and tends to increase distance from the proxy reference.

See `qa_refined_proxy_controller_v1/summary.json` for machine-readable metrics and `final_report.md` for the interpreted milestone report.
