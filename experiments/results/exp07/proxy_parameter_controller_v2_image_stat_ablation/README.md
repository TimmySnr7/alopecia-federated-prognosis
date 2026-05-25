# Experiment 07 v2 Image-Statistic Ablation

This ablation identifies which image statistics drive the image-aware controller result in Experiment 07 v2.

## Result

The full nine-statistic model is not the best parameter predictor. A sparse subset performs substantially better.

| Feature set | Val MAE | Test MAE | Test normalized MAE |
|---|---:|---:|---:|
| No image statistics | 0.0101 | 0.0229 | 0.0909 |
| All nine image statistics | 0.0110 | 0.0084 | 0.1240 |
| Hair-presence index only | 0.0035 | 0.0023 | 0.0423 |
| Hair-presence + masked gray mean | 0.0032 | 0.0022 | 0.0156 |
| Hair-presence + masked gray mean + mask area | 0.0032 | 0.0022 | 0.0154 |
| Hair-presence + masked gray mean + mask area + texture p90 | 0.0032 | 0.0022 | 0.0148 |

## Interpretation

Most of the useful image-aware signal is concentrated in two to four statistics:

- hair-presence index,
- masked grayscale mean,
- mask-area fraction,
- masked texture 90th percentile.

Adding all nine features hurts normalized error, likely because the training set has only 42 rows and the extra correlated features overfit or destabilize the ridge solution.

This means the best next controller is not a larger feature vector by default. It should use a sparse, interpretable image-statistic set and add explicit monotonicity or QA refinement.
