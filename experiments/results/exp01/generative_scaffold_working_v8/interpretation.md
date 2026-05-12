# Interpretation

## High-level assessment

`v8` is an important negative-but-informative result.

Replacing the scaffold's weak internal severity head with a frozen external
severity scorer improved the conceptual validity of the supervision signal, but
it still did not produce convincing target-dependent severity sweeps.

## What it supports

This run supports the claim that:

- external scorer guidance is technically integrated and trainable,
- the current scaffold remains numerically stable under stronger auxiliary
  supervision,
- and the project has moved beyond purely architectural conditioning tweaks.

## What it does not support

This run does **not** support the claim that the current scaffold learns
meaningful severity-conditioned transformations on the public `exp01_working`
subset.

The sweep outputs remained too similar across target severities, indicating
that stronger supervision alone was not enough when targets were sampled across
the full severity range.

## Why `v9` follows from this

`v9` narrows the target-shift problem to adjacent severity transitions. This is
both more clinically plausible for early proxy-prognosis scaffolding and more
learnable than random jumps such as `1 -> 7` on a tiny heterogeneous dataset.
