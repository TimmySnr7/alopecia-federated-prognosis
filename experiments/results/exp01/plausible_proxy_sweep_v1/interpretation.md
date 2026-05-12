# Interpretation

## High-level assessment

`plausible_proxy_sweep_v1` is the first usable visual plausibility guardrail for
`exp01`.

The sweep produces a readable severity direction: lower targets preserve and
emphasize visible stubble, while higher targets progressively smooth and
suppress that texture. The frozen scorer's expected severity is monotonic across
all targets.

## What it supports

This result supports the claim that:

- visual plausibility constraints can prevent the color and texture exploits
  seen in scorer-guided residual optimization,
- the frozen scorer is sensitive to a plausible monotonic texture/smoothing
  proxy,
- and future learned generators should be evaluated against both target-control
  metrics and visual-change guardrails.

## What it does not support

This is not a learned prognostic generator and should not be interpreted as
patient-specific hair regrowth or progression synthesis.

The transform is deterministic and hand-designed. It validates the direction of
the visual pipeline, not the final modelling approach.

## Practical conclusion

Keep this run as a plausibility baseline and regression test. A future learned
model should beat it by producing target-separated, naturalistic edits without
collapsing into classifier-only artifacts.
