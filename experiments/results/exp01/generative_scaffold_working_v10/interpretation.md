# Interpretation

## High-level assessment

`v10` is a failed rebalance attempt.

Although it reduced the explicit severity-guidance weight, the resulting sweep
collapsed even more aggressively than `v9`, producing nearly uniform
purple-gray outputs across target severities.

## What it supports

This run supports the claim that:

- simple loss-weight rebalancing alone is not enough for the current
  whole-image scaffold,
- the frozen scorer can still dominate the image path in undesirable ways,
- and a more structural change is needed to preserve source appearance.

## What it does not support

This run does **not** support treating the current whole-image reconstruction
formulation as the right base for the next experiments.

It also does **not** support using the `v10` sweep as evidence of meaningful
conditioned severity control.

## Why `v11` follows from this

`v11` pivots from direct full-image prediction to bounded residual prediction.
That anchors the model to the source image and makes condition-driven edits act
more like small appearance changes rather than unconstrained full-frame
regeneration.
