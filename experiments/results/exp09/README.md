# Experiment 09 Results

Experiment 09 is the final fragility-aware evaluation of the alopecia severity-synthesis pipelines. It stress-tests the fixed Experiment 08 validation/test split under realistic perturbations and reports a composite fragility score:

`F(P,e) = 0.4(1 - S_order) + 0.4(1 - S_QA) + 0.2 E_crit`

Lower values indicate lower fragility.

The executable result bundle is in `fragility_aware_evaluation_v1/`.

## Pipelines

- `P1_proxy`: QA-gated deterministic proxy baseline.
- `P2_diffusion_archived_exp06`: best archived diffusion editor from Experiment 06.
- `P3_sparse2_ridge`: sparse2 controller using hair-presence index and masked grayscale mean.
- `P4_sparse4_ridge`: sparse4 controller using sparse2 plus mask area fraction and texture p90.

## Boundary

P1, P3, and P4 were re-rendered under perturbations. P2 is included as an archived Exp06 comparator because a lightweight deterministic diffusion inference artefact is not available for per-perturbation reruns.
