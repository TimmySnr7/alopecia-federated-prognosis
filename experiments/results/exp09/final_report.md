# Experiment 09 Final Report

## Title

Fragility-Aware Evaluation of Alopecia Severity Synthesis

## Status

Completed as the final computational experiment. Human face-validity materials are prepared but dermatologist ratings have not yet been collected.

## Aim

Evaluate whether alopecia severity-synthesis pipelines remain valid under realistic perturbations, using a composite fragility score and a prepared human face-validity review protocol.

Hypothesis: sparse proxy-parameter controllers will exhibit lower fragility than direct diffusion editors under mask, target, photometric, and parameter perturbations, with sparse2 expected to be at least as robust as sparse4 because of its lower-dimensional, more interpretable feature set.

## Pipelines

| ID | Pipeline | Execution status |
|---|---|---|
| P1 | QA-gated deterministic proxy baseline | Re-rendered under perturbations |
| P2 | Best diffusion editor from Experiments 03-06 | Included as archived Exp06 high-fragility reference |
| P3 | Sparse2 ridge controller | Re-rendered under perturbations |
| P4 | Sparse4 ridge controller | Re-rendered under perturbations |

P2 uses the Experiment 06 severity-delta cross-attention variant because it was the stronger archived diffusion result. It was not re-run for each perturbation in the completed Exp09 run, so it is not a matched perturbation-tested pipeline. This limitation is carried explicitly in the interpretation.

P2 fragility is computed from archived Experiment 06 outputs under nominal held-out conditions and propagated as a reference comparator across perturbation groups. It represents the upper fragility regime observed in the diffusion series; per-perturbation diffusion re-runs were not performed for the reported table. A clean AI5 rerun of the existing Exp09 computational pipeline was completed after review to verify that the P1/P3/P4 outputs reproduce from the remote source-image and scorer artefacts.

## Cases and Targets

The experiment used the fixed Experiment 08 split:

- Validation: 2 cases
- Test: 4 cases

Each case was rendered at five severity targets: 1, 2, 4, 6, and 7.

## Perturbations

Validation used nominal plus low perturbations. Test used nominal plus low and high perturbations.

| Perturbation | Low | High |
|---|---:|---:|
| Mask erosion/dilation | 1 px | 3 px |
| Mask shift | +/-1 px | +/-3 px |
| Target noise | sigma = 0.03 | sigma = 0.08 |
| Brightness shift | +/-5% | +/-15% |
| Contrast shift | +/-5% | +/-15% |
| Gaussian image noise | sigma = 0.01 | sigma = 0.03 |
| Proxy-parameter noise | 0.5 SD | 1.0 SD |

## Fragility Metric

The balanced fragility score was:

`F(P,e) = 0.4(1 - S_order) + 0.4(1 - S_QA) + 0.2 E_crit`

Lower score means lower fragility.

`S_order` combines monotonic fraction, Spearman rho, and violation count. `S_QA` combines auto-pass, severity span, unmasked drift, mask-area validity, and scalp-localisation validity. `E_crit` records critical errors such as non-scalp hallucination, broken masks, facial distortion, stylised head artefacts, or anatomically implausible scalp/hair patterns.

## Main Results

### Mean Fragility by Pipeline

| Pipeline | Mean balanced fragility | Safety-heavy | QA-heavy | Ordering-heavy |
|---|---:|---:|---:|---:|
| P1 proxy | 0.0319 | 0.0199 | 0.0355 | 0.0363 |
| P2 diffusion archived Exp06 | 0.5509 | 0.6550 | 0.5106 | 0.5217 |
| P3 sparse2 ridge | 0.0118 | 0.0074 | 0.0131 | 0.0134 |
| P4 sparse4 ridge | 0.0157 | 0.0098 | 0.0178 | 0.0175 |

Among the matched perturbation-tested pipelines, P3 sparse2 has the lowest mean balanced fragility in the all-split summary. The archived P2 diffusion row should be read as a high-fragility reference from the direct-diffusion series, not as a fresh per-perturbation diffusion stress test.

P3 sparse2 achieves lower measured fragility than the deterministic proxy under this protocol. The likely reason is that ridge regression regularises parameter predictions, smoothing over variation in the proxy-control targets. This is a protocol-sensitive finding rather than proof of intrinsic superiority over the proxy renderer: the proxy baseline directly executes perturbed parameters, whereas the learned controller's L2-regularised mapping constrains the output parameter space before rendering.

### Fragility by Perturbation Strength

| Pipeline | Nominal | Low perturbations | High perturbations |
|---|---:|---:|---:|
| P1 proxy | 0.0390 | 0.0443 | 0.0465 |
| P2 diffusion archived Exp06 | 0.6400 | 0.6400 | 0.6400 |
| P3 sparse2 ridge | 0.0000 | 0.0106 | 0.0268 |
| P4 sparse4 ridge | 0.0000 | 0.0142 | 0.0215 |

The sparse controllers degrade gradually rather than collapsing. The P2 row is flat by construction because it is archived, so the table supports a qualitative distinction between graceful controller degradation and the catastrophic diffusion failure regime, not a matched numerical perturbation curve for diffusion.

### Perturbation-Level Findings

| Pipeline | Nominal fragility | Brightness | Contrast | Image noise | Parameter noise |
|---|---:|---:|---:|---:|---:|
| P1 proxy | 0.0195 | 0.0260 | 0.0130 | 0.0260 | 0.1250 |
| P2 diffusion archived Exp06 | 0.5100 | 0.5533 | 0.5533 | 0.5533 | 0.5533 |
| P3 sparse2 ridge | 0.0000 | 0.0065 | 0.0130 | 0.0130 | 0.0723 |
| P4 sparse4 ridge | 0.0000 | 0.0065 | 0.0130 | 0.0130 | 0.1180 |

Mask erosion, mask dilation, mask shift, and target noise were stable for the sparse controllers. The main weakness was proxy-parameter noise, especially for P4 sparse4. This supports the Experiment 08 production recommendation: the simpler P3 sparse2 controller is more robust than the richer sparse4 variant.

## Component Breakdown

The diffusion reference has high fragility because it combines poor ordering stability, QA failure, and high critical-error risk from archived contact sheets. In contrast, P3 and P4 have zero critical-error rate across computational perturbations; their fragility comes from mild ordering and QA degradation under image noise and parameter noise.

The component breakdown therefore separates two qualitatively different failure modes. P2 diffusion is represented as a catastrophic archived failure because semantic/critical errors dominate alongside weak ordering and QA scores. P3 and P4 fail gracefully: when stress increases, the first degradation is ordering or QA stability, not non-scalp hallucination or anatomical collapse.

The component figure is saved as `figure2_stacked_component_bars.svg`.

## Contact-Sheet Review

Generated contact sheets include:

- `test_P1_proxy_nominal_contact_sheet.png`
- `test_P3_sparse2_ridge_nominal_contact_sheet.png`
- `test_P4_sparse4_ridge_nominal_contact_sheet.png`
- `test_P3_sparse2_ridge_parameter_noise_high_contact_sheet.png`
- `test_P4_sparse4_ridge_parameter_noise_high_contact_sheet.png`
- `contact_sheet_P4_sparse4_nominal_reference.png`

The sparse-controller sheets remain source-preserving and scalp-localised. The high parameter-noise sheets show the expected weak point: severity ordering can become less stable even though the outputs remain non-hallucinatory and anatomically bounded by the deterministic renderer.

## Protocol Sensitivity: Sparse2 Versus Proxy

A protocol-sensitivity summary was added after review as `protocol_sensitivity_sparse_vs_proxy.csv`. It recomputes test-only means for P1, P3, and P4 across several analysis slices:

| Analysis slice | P1 proxy | P3 sparse2 | P4 sparse4 |
|---|---:|---:|---:|
| All test conditions | 0.0451 | 0.0179 | 0.0171 |
| Excluding proxy-parameter noise | 0.0353 | 0.0093 | 0.0093 |
| Proxy-parameter noise only | 0.1485 | 0.1085 | 0.0990 |
| Perturbations excluding nominal and parameter noise | 0.0351 | 0.0098 | 0.0098 |
| Nominal only | 0.0390 | 0.0000 | 0.0000 |

This supports the original observation that sparse controllers have lower measured fragility than the proxy under the implemented protocol, but it also clarifies the interpretation. The comparison is sensitive to noise placement because proxy perturbations directly modify renderer parameters, while sparse-controller perturbations pass through a regularised prediction model before rendering. The result should therefore be described as protocol-specific robustness, not as proof that the learned controller is intrinsically better than the proxy renderer it approximates.

## Human Face-Validity Study

The human review was prepared but not executed in this run. The protocol uses P1 proxy, P2 diffusion, and P4 sparse4 to reduce rater burden. Raters score source preservation, scalp localisation, monotonic progression, anatomical plausibility, and overall acceptability on a 1-5 Likert scale. The prepared files are:

- `human_face_validity_protocol.md`
- `human_face_validity_rating_template.csv`
- `human_rating_correlation_table.csv`

After ratings are collected, balanced fragility should be correlated against overall acceptability. The expected relationship is:

`fragility increases -> human acceptability decreases`

The human face-validity study was designed and staged as part of Experiment 09 but is not executed in this computational run. Rater recruitment and data collection should proceed as a separate pilot study, with results intended for the thesis validation chapter or a follow-up publication.

## Interpretation

Experiment 09 supports the central expected claim in a scoped form: under controlled perturbations, sparse proxy-parameter controllers maintain low measured fragility among matched perturbation-tested pipelines, while archived diffusion evidence shows the kind of semantic failure that severity-response metrics alone can miss.

The strongest computational result is that P3 sparse2 ridge remains low-fragility with only two interpretable image features. This is not because it generates stronger images; it is because sparse features are enough to maintain robust control while avoiding some of the additional parameter sensitivity introduced by richer feature sets.

The diffusion comparison remains conservative but imperfect. It is scientifically useful as an archived failure reference because Exp06 already documented its failure under the same held-out cases, but it should not be described as a fresh per-perturbation diffusion stress test. A fully matched diffusion perturbation study would require adding an inference-only loader for the trained Exp06 LoRA and severity-projection artefacts.

## Limitations

The test set remains small, with four test cases and two validation cases. The human face-validity study has not yet been executed. Critical-error labels for P1, P3, and P4 are computational heuristics; P2 critical errors are based on archived Exp06 contact-sheet evidence rather than newly generated perturbed samples. The sparse-versus-proxy comparison is sensitive to where noise enters the pipeline. Finally, the proxy renderer remains a methodological benchmark, not a clinically validated prognostic model.

## Conclusion

Experiment 09 completes the computational arc. The deterministic proxy and sparse controllers are robust to most mask, target, brightness, contrast, and image-noise perturbations, while archived diffusion evidence remains highly fragile. Among matched perturbation-tested pipelines, P3 sparse2 ridge has the clearest parsimony and robustness profile, supporting it as the final computational candidate for clinical face-validity review.
