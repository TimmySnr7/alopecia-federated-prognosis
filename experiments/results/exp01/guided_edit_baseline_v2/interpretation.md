# Interpretation

## High-level assessment

`guided_edit_baseline_v2` is the first clearly meaningful controllability result
for `exp01`.

The frozen severity scorer predicts the intended target class for every edited
output, and both the discrete predicted severity sequence and the expected
severity sequence are perfectly monotonic.

## What it supports

This result supports the claim that:

- the current scorer/data stack contains enough signal to support explicit
  target-directed editing,
- meaningful severity control is achievable on the present public subset,
- and the failures of the learned scaffolds should not be interpreted as
  evidence that the conditioning problem is impossible.

## What it does not support

This result does **not** yet support claiming clinically plausible prognosis
generation.

The outputs remain visually stylized and somewhat adversarial in texture, so
this should be treated as a scorer-guided editing baseline rather than a final
generative model.

## Practical conclusion

This run is the first auditable positive control for Experiment 1. It should be
used as:

- a sanity-check baseline for target controllability,
- a reference for later learned models,
- and evidence that the bottleneck now lies in visual plausibility and
  naturalistic generation, not in target conditioning alone.
