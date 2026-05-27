# Experiment 09 Diffusion Perturbation Rerun

This directory contains source-safe tabular outputs from the matched perturbation rerun of the Experiment 06 severity-delta diffusion editor.

The rerun was executed on AI5 using:

- saved Exp06 LoRA weights: `exp06_severity_delta_cross_attention_v1/unet_lora_state_dict.pt`
- saved Exp06 severity projection: `exp06_severity_delta_cross_attention_v1/severity_projection.pt`
- model: `runwayml/stable-diffusion-v1-5`
- ControlNet: `lllyasviel/control_v11p_sd15_softedge`
- scorer: `exp05_proxy_severity_scorer_v1.ckpt`
- image size: 256
- inference steps: 20

The full image contact sheets were generated on AI5 but are not included in the source-safe repository because they contain third-party source-image derivatives. The CSV and JSON files here are sufficient to reproduce the measured P2 rows merged into `../fragility_aware_evaluation_v1/`.

Critical-error values are computational critical-risk heuristics based on scalp localisation and masked L1 to proxy. They should be checked against the generated contact sheets before being treated as final human semantic labels.
