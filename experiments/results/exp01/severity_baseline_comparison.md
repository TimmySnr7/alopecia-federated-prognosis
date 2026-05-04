# Experiment 1 Severity Baseline Comparison

## Runs compared

- `top_only_v1`
- `top_priority_v1`
- `top_priority_v2`

## Summary judgement

`top_only_v1` is useful only as a clean loader and smoke-test subset. It is too
small to support meaningful validation.

`top_priority_v1` is the first subset/run combination that provides a usable
validation signal.

`top_priority_v2` is the current best baseline because pretrained features,
augmentation, and class weighting improve stability sufficiently to justify
formalising the result.

## Why `top_priority_v2` wins

- best observed validation accuracy (`0.333`)
- substantially more stable validation behaviour than `top_only_v1`
- better training dynamics than the earlier unrefined baseline
- enough heterogeneity to act as the current working subset for the next
  centralised generative scaffold

## Caveat

This is still a feasibility-stage result under severe data limitations. It
should be cited as an early baseline and pipeline milestone, not as a
benchmark-quality predictive model.
