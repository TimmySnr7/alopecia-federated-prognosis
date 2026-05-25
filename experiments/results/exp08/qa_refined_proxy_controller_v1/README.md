# Experiment 08 v1: QA-Refined Proxy Controller

This run was executed on AI5 using:

```bash
python experiments/scripts/run_exp08_qa_refined_proxy_controller.py \
  --scorer-checkpoint /home/tmushuru/datasets/alopecia_public/metadata/exp05_proxy_severity_scorer_v1.ckpt \
  --output-dir experiments/results/exp08/qa_refined_proxy_controller_v1 \
  --iterations 30
```

## Variants

- `sparse2_ridge`: image-stat ridge using hair-presence index and masked grayscale mean.
- `sparse4_ridge`: image-stat ridge using hair-presence index, masked grayscale mean, mask area fraction, and texture p90.
- `sparse4_refined_lambda_*`: bounded post-prediction QA refinement initialised from `sparse4_ridge`, with monotonicity weights 0.5, 1.0, 5.0, and 10.0.

## Key Artefacts

- `summary.json`: full per-model and per-case metrics.
- `metrics_summary.csv`: compact tabular summary for manuscript/thesis use.
- `exp08_tradeoff_curve.svg`: test proxy-L1 vs. monotonicity plot.
- `test_sparse4_ridge_contact_sheet.png`: recommended contact sheet for visual reporting.
- `test_sparse2_ridge_contact_sheet.png`: most parsimonious sparse controller contact sheet.

## Headline Result

Both unrefined sparse controllers achieved 100% validation and test auto-pass, 1.000 monotonicity, zero monotonicity violations, and Spearman rho of 1.000. This resolves the Experiment 07 v2 image-stat monotonicity inversion without needing the bounded refinement loop.
