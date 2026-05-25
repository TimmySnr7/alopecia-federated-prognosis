# Interpretation

## High-level assessment

`proxy_residual_editor_v2` is a decisive negative result for the current
Experiment 03 residual-editor formulation.

The v2 changes directly targeted the v1 failure mode by increasing the cost of
large source-to-target severity under-response and by matching the proxy target
in scorer space. This did not recover held-out target separation. The learned
editor remained visually plausible in many panels, but the scorer-visible
severity trajectory stayed collapsed, especially on test cases.

## What changed from v1

The v2 run added:

- severity-delta-weighted per-sample losses,
- dataset-balanced weighted sampling,
- stronger masked proxy and residual-delta reconstruction,
- increased residual scale (`0.35` to `0.5`),
- scorer probability-distribution consistency against the proxy target,
- and expected-severity consistency against the proxy target.

These are the most direct low-complexity fixes for the v1 under-conditioning
failure.

## What worked

- Training completed on AI5 using the QA-gated Experiment 02 manifest.
- The model remained mask-localized enough to avoid gross background leakage.
- Validation unmasked drift remained below the QA threshold:
  `0.00199 <= 0.01`.
- Contact sheets do not show the severe global colour collapse seen in the
  earlier Experiment 01 learned scaffolds.

## What failed

The learned editor still does not beat the proxy baseline.

Validation learned auto-pass rate stayed at `0 / 2`, while the proxy baseline
passed `2 / 2`. Test learned auto-pass rate fell to `0 / 4`, while the proxy
baseline passed `4 / 4`.

The primary failure is still target-severity collapse:

- Validation learned mean expected span: `0.300`
- Validation proxy mean expected span: `1.816`
- Test learned mean expected span: `0.043`
- Test proxy mean expected span: `2.276`

Compared with v1, v2 did not improve the scientific gate. It increased
pair-level proxy error and test unmasked drift while reducing test expected
span from `0.301` to `0.043`.

The likely mechanism is overfitting or loss dominance on an extremely small
validation regime: the stronger weighted and scorer-space losses preserved the
two-case validation span at roughly the v1 level, but they encouraged a
source/proxy residual template that did not generalise to the four held-out
test cases.

## Final Experiment 03 conclusion

Experiment 03 should be closed as a negative learned-editor experiment.

Across v1 and v2, the learned residual editor can preserve visual plausibility
and localization, but it cannot reliably learn target-conditioned severity
separation from the current tiny QA-gated proxy-pair pool. The failure is not a
simple missing loss-weight issue: direct delta weighting, balanced sampling, and
scorer-space supervision all failed to recover the proxy baseline.

The strongest supported conclusion is that Experiment 02's proxy pipeline is a
usable QA-gated weak-supervision and evaluation baseline, but the current
Experiment 03 residual editor is not a suitable final prognosis-generation
model.

## Recommended next step

Do not continue tuning this exact residual-editor setup as Experiment 03.

The next learned-editing attempt should be framed as a new experiment. The
primary path should be a pretrained image-editing backbone with mask-localized
adaptation, because that directly supports the proposal's Contribution C1
latent conditional diffusion architecture and the Experiment 02 supervisory
recommendation to pivot to latent conditional diffusion. Supporting directions
include:

- generate more QA-passing pseudo-pairs before learning,
- train a model to predict normalized proxy residuals with explicit target
  endpoints,
- add identity/source reconstruction and target-contrastive objectives across
  same-source severity sweeps,
- and use the QA-gated proxy pairs as weak supervision for the pretrained
  latent editing model.

For the thesis/project record, Experiment 03 provides evidence that small-data
learned residual editing is not yet reliable under the current public-data
constraints.
