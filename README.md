# Alopecia Federated Prognosis

Research workspace for the PhD project on privacy-preserving generative prognosis for alopecia.
The repository is designed around four core goals:

1. Build a reproducible latent-diffusion baseline for alopecia prognosis.
2. Extend that baseline into a feasibility-scoped federated learning pipeline.
3. Evaluate fidelity, calibration, fairness, privacy, and human perception consistently.
4. Keep student contributions optional, sandboxed, and easy to review before promotion into the core codebase.

## Repository Principles

- `models/`, `evaluation/`, `data/`, and `experiments/` contain the core research code.
- `student_projects/` contains isolated sandboxes for BSc/MSc contributors.
- `docs/` stores governance, proposal snapshots, and dataset-audit artefacts.
- `thesis_outputs/` stores publication-ready figures, tables, and preprint material derived from tracked experiments.

This structure is intentionally self-contained: if no student project is completed, the core
workspace remains usable for the PhD itself.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Typical Workflow

1. Add or update an experiment configuration in `experiments/configs/`.
2. Confirm dataset entries and licence notes in `data/dataset_registry.yaml`.
3. Record protocol assumptions or audit notes under `docs/` before the first run.
4. Run the appropriate driver in `experiments/scripts/`.
5. Save generated artefacts to `experiments/results/` using DVC or Git LFS.
6. Generate reusable reports and thesis assets under `evaluation/reports/` and `thesis_outputs/`.

## Experiment 1 Data Manifests

For `exp01`, keep raw public datasets outside Git and build pooled manifests from
an external data root:

```bash
python experiments/scripts/build_exp01_manifest.py \
  experiments/configs/exp01_centralised_baseline.yaml \
  --data-root ~/datasets/alopecia_public
```

This writes:
- `~/datasets/alopecia_public/metadata/exp01_master_index.csv`
- `~/datasets/alopecia_public/manifests/exp01_train.csv`
- `~/datasets/alopecia_public/manifests/exp01_val.csv`
- `~/datasets/alopecia_public/manifests/exp01_test.csv`

For a first loader smoke test on the cleaner top-view subset:

```bash
python experiments/scripts/smoke_test_exp01_loader.py \
  ~/datasets/alopecia_public/manifests/exp01_top_only_train.csv \
  --batch-size 4
```

For a first trainable severity baseline smoke test:

```bash
python experiments/scripts/train_severity_baseline.py \
  ~/datasets/alopecia_public/manifests/exp01_top_only_train.csv \
  ~/datasets/alopecia_public/manifests/exp01_top_only_val.csv \
  --epochs 3 \
  --batch-size 4
```

For a slightly stronger early baseline on the larger top-priority subset:

```bash
python experiments/scripts/train_severity_baseline.py \
  ~/datasets/alopecia_public/manifests/exp01_top_priority_train.csv \
  ~/datasets/alopecia_public/manifests/exp01_top_priority_val.csv \
  --epochs 10 \
  --batch-size 4 \
  --pretrained \
  --augment \
  --class-weighting
```

For the first conditional generative-batch smoke test:

```bash
python experiments/scripts/smoke_test_exp01_generative_batch.py \
  ~/datasets/alopecia_public/manifests/exp01_working_train.csv \
  --batch-size 4
```

For the first centralised generative training scaffold:

```bash
python experiments/scripts/train_exp01_generative.py \
  experiments/configs/exp01_centralised_baseline.yaml \
  ~/datasets/alopecia_public/manifests/exp01_working_train.csv \
  ~/datasets/alopecia_public/manifests/exp01_working_val.csv \
  --epochs 3 \
  --batch-size 4 \
  --checkpoint-path ~/datasets/alopecia_public/metadata/exp01_working_generative_scaffold.ckpt
```

For a first conditioning-sensitivity sweep on one fixed image:

```bash
python experiments/scripts/run_exp01_severity_sweep.py \
  ~/datasets/alopecia_public/metadata/exp01_working_generative_scaffold.ckpt \
  /path/to/example_top_view.png \
  --output-path ~/datasets/alopecia_public/metadata/exp01_working_severity_sweep.png
```

## Student Contribution Model

Each directory in `student_projects/` is a sandbox with its own README and handover
expectations. Nothing moves into the core codebase until it has:

- a reproducible script or module,
- a short technical handover,
- a config or protocol file,
- and a successful review.

## Immediate Operational Setup

- `CONTRIBUTING.md` defines coding, review, and handover rules.
- `.github/workflows/ci.yml` provides a lightweight CI entry point.
- Branch protection is enabled on `main` for safer collaboration.
- `ROADMAP.md` tracks the phased build-out of the research programme.

## Citation

If you reuse this repository structure, code, or experimental design, please cite the
repository metadata in [CITATION.cff](./CITATION.cff). A simple BibTeX entry is:

```bibtex
@software{musharu_alopecia_federated_prognosis_2026,
  author = {Musharu, Timothy},
  title = {Alopecia Federated Prognosis},
  year = {2026},
  url = {https://github.com/TimmySnr7/alopecia-federated-prognosis},
  license = {Apache-2.0}
}
```

Where appropriate, also cite the associated PhD proposal, preprints, and future publications
that emerge from the repository.
