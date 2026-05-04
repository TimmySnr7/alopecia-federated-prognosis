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
2. Run the appropriate driver in `experiments/scripts/`.
3. Save generated artefacts to `experiments/results/` using DVC or Git LFS.
4. Generate reusable reports and thesis assets under `evaluation/reports/` and `thesis_outputs/`.

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
