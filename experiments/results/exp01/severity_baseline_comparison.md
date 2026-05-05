# Experiment 1 Severity Baseline Comparison

## Runs compared

- `top_only_v1`
- `top_priority_v1`
- `top_priority_v2`
- `working_v3`

## Summary judgement

`top_only_v1` is useful only as a clean loader and smoke-test subset. It is too
small to support meaningful validation.

`top_priority_v1` is the first subset/run combination that provides a usable
validation signal.

`top_priority_v2` is the current best baseline because pretrained features,
augmentation, and class weighting improve stability sufficiently to justify
formalising the result.

`working_v3` becomes the final classification-side checkpoint because it keeps
the stronger recipe while adding richer class-level evaluation on the formal
`exp01_working` subset.

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
## Transition note

The severity-baseline series established that the current public-data subset is trainable and that `exp01_working` is the right development subset. The next milestone is therefore the centralised generative scaffold recorded in `generative_scaffold_working_v1/`, which marks the transition from classification-side validation into the actual generative path.
