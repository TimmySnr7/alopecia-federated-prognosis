# Interpretation

## High-level assessment

`source_aware_proxy_batch_v1` improves the auditability of visual plausibility
without over-claiming model performance.

The source-aware mask preset did not increase the headline auto-pass rate above
Experiment 01's `13 / 15`, but it made the failure mode cleaner. The five
`unidpro_hair_loss_male_norwood_scale` cases now use a smaller dataset-specific
top-down mask with area fraction `0.246`, and the two remaining failures have
low unmasked drift. This suggests the remaining issue is weak target separation
for those cases rather than uncontrolled background editing.

## What it supports

This result supports three practical Experiment 02 claims:

- The severity-scorer entry gate has been met for engineering QA.
- Source-aware mask selection is implemented and recorded per case.
- A QA-gated pseudo-pair manifest can be exported without including failed
  cases.

## What it does not support

This does not prove that a learned residual editor will generalize.

The curated pseudo-pair manifest contains `78` severity-shift rows from only
`13` source images. That is enough to test pipeline mechanics and weak
supervision plumbing, but not enough to claim a robust learned generator.

## Practical conclusion

Experiment 02 is in good shape as a localization, scorer-gate, and curated
proxy-supervision experiment.

The next learned-model run should use `proxy_pair_manifest.csv` as an explicit
QA-gated supervision input. It should not train on the two failed cases unless
their masks and scorer response are manually corrected and re-evaluated.
