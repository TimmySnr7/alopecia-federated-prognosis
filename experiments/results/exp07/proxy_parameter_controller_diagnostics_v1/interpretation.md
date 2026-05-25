# Experiment 07 Diagnostic Interpretation

## Verdict

The suspiciously low Experiment 07 metadata-ridge error is not caused by source-case split leakage or by obvious feature leakage. It is caused by a highly deterministic, low-variance proxy parameter space.

This means the result is real but should be framed conservatively.

## Split Integrity

The source-case split is clean:

- train cases: 7
- validation cases: 2
- test cases: 4
- train/validation overlap: none
- train/test overlap: none
- validation/test overlap: none

The metadata model does not receive source case IDs or target parameter values. Its features are:

- intercept,
- normalized source severity,
- normalized target severity,
- normalized severity delta,
- absolute severity delta,
- lower-severity direction scalar,
- higher-severity direction scalar,
- dataset-key one-hot variables,
- label-schema one-hot variables.

## Why the Error Is Near Zero

The target vector is not a rich learned parameter space yet. It is the current effective proxy-control vector. Several components are constant:

- `ellipse_cx`,
- `mask_blur_radius`,
- `texture_blur_radius`.

The ellipse shape has only two unique presets for `ellipse_cy`, `ellipse_rx`, and `ellipse_ry`. The remaining edit controls are severity-scaled versions of global proxy constants. Therefore, source severity, target severity, and dataset/schema metadata are almost sufficient to reconstruct the entire target vector.

## Shuffle Controls

The shuffled controls confirm that the evaluation is not trivially broken:

| Model | Original test MAE | Test MAE against shuffled truth | Test MAE after shuffled train targets |
|---|---:|---:|---:|
| Severity ridge | 0.00487 | 0.13922 | 0.09933 |
| Metadata ridge | 0.000007 | 0.13436 | 0.08958 |

If the result were caused by direct leakage or a broken error calculation, the shuffled controls would remain near zero. They do not.

## Revised Claim

The correct claim is:

> Experiment 07 demonstrates that the current QA-gated proxy control policy is learnable and automatable with a simple metadata-aware controller.

The incorrect claim would be:

> Experiment 07 demonstrates that the model learned clinically meaningful image-dependent progression.

The current result is a useful positive computational result, but it is also evidence that the existing proxy parameter space is too simple to test the value of image-aware learning.

## Recommendation

Close Experiment 07 as a positive automation result only after adding this diagnostic caveat to the final report. Then design Experiment 07 v2 or Experiment 08 around a richer parameter space, where image appearance can plausibly matter.
