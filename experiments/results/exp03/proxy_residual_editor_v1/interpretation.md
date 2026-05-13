# Interpretation

## High-level assessment

`proxy_residual_editor_v1` is a useful negative result.

The model learned to keep edits mask-localized and background-safe, but it did
not learn enough target-conditioned severity separation. It approximates proxy
images with low average L1 error, yet the scorer-visible severity span collapses
relative to the proxy baseline.

## What worked

- The training run completed on the QA-gated Experiment 02 manifest.
- Background/unmasked drift stayed low: validation mean max unmasked delta was
  `0.000221`, and test mean max unmasked delta was `0.001750`.
- Pair-level proxy reconstruction was numerically close: final validation proxy
  L1 was `0.00320`, and final test proxy L1 was `0.00717`.
- Contact sheets show no global colour collapse or obvious adversarial texture
  shortcut like the earlier Experiment 01 scorer-guided learned scaffolds.

## What failed

The learned editor does not beat the proxy baseline.

Validation learned auto-pass rate was `0 / 2`, while the proxy baseline passed
`2 / 2`. Test learned auto-pass rate was `1 / 4`, while the proxy baseline
passed `4 / 4`.

The main failure mode is severity-span compression:

- Validation learned mean expected span: `0.291`
- Validation proxy mean expected span: `1.816`
- Test learned mean expected span: `0.301`
- Test proxy mean expected span: `2.276`

The learned outputs are visually plausible but too conservative. They often
look like softened source-preserving averages rather than distinct target
severity edits.

## Practical conclusion

Experiment 03 v1 should be treated as a diagnostic baseline, not a solved learned
editor.

The next run should explicitly counter under-conditioning. Reasonable options
include stronger target-delta weighting, target-span or scorer-consistency loss,
case-balanced sampling, more aggressive augmentation, and/or directly predicting
the proxy residual with target-delta normalization rather than reconstructing the
full proxy image.
