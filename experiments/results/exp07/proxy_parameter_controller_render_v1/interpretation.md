# Experiment 07 Render v1 Interpretation

## Verdict

Experiment 07 is the first learned approach after Experiment 02 to recover the core computational QA behavior of the proxy baseline.

Both the severity-only ridge controller and the metadata-aware ridge controller achieved 100% auto-pass on validation and test after rendering through the deterministic proxy renderer. This is a materially different outcome from Experiments 03--06: the learned component controls a constrained parameter space rather than generating image pixels directly, so the hallucination pathway is removed by construction.

## Quantitative Result

| Model | Validation auto-pass | Test auto-pass | Validation monotonicity | Test monotonicity | Validation span | Test span | Test mean max unmasked drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Constant mean | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.0020 |
| Severity ridge | 1.000 | 1.000 | 1.000 | 1.000 | 5.651 | 5.346 | 0.0055 |
| Metadata ridge | 1.000 | 1.000 | 1.000 | 1.000 | 4.458 | 5.349 | 0.0061 |

The constant baseline fails because it produces no severity span. The learned controllers pass because they recover the severity-direction controls of the proxy renderer.

## Comparison With Proxy Reference

The metadata-aware controller closely reproduces the Experiment 02/06 proxy reference:

- validation span: `4.458`, matching the calibrated proxy span of approximately `4.458`;
- test span: `5.349`, matching the calibrated proxy span of approximately `5.350`;
- unmasked drift remains below the `0.01` QA threshold on both splits;
- mask-area fractions match the source-aware proxy masks.

The severity-only ridge controller also passes both splits, but its validation mask area is approximately `0.516`, whereas the metadata-aware controller recovers the validation mask area of `0.246`. This confirms that dataset/source metadata is needed to recover source-aware mask presets, while severity variables are sufficient for the edit-strength controls.

## Scientific Meaning

This result supports the thesis pivot proposed after Experiment 06. The controllable part of the successful synthesis mechanism is low-dimensional and learnable. Instead of asking a general diffusion model to learn scalp anatomy, severity direction, and source preservation simultaneously, Experiment 07 learns only the control policy and delegates rendering to an auditable deterministic mechanism.

The outcome changes the experimental arc:

- Experiments 03--06 ruled out direct learned image generation at the current data scale.
- Experiment 07 shows that learned control over the validated proxy mechanism is feasible.

This is a positive computational result and a defensible next C1 artefact direction.

## Limitations

The result is partly circular. The metadata-aware controller succeeds because the current proxy parameterisation is strongly deterministic and dataset-specific. It should therefore be framed as learning to reproduce and automate the validated proxy control policy, not as learning clinical progression from real longitudinal data.

The experiment also does not yet prove that image features improve control. The current successful models use severity and metadata only. An image-aware CNN+MLP controller is justified only if the proxy parameter space is expanded or if residual parameter-choice errors remain after metadata conditioning.

## Next Step

Experiment 07 v2 should test a richer proxy parameter space and compare:

1. severity-only controller,
2. metadata-aware controller,
3. image-aware CNN+MLP controller,
4. optional QA-refined controller.

The decisive question is whether image-aware control improves held-out QA or parameter selection beyond metadata-aware rules.
