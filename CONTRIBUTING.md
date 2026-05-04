# Contributing

## Ground Rules

- Work on a feature branch, never directly on `main`.
- Keep every contribution reproducible from code plus config.
- Student sandbox work belongs in `student_projects/` until reviewed.
- Core code belongs in `data/`, `models/`, `evaluation/`, or `experiments/`.

## Coding Style

- Format Python with `black`.
- Use type hints for public functions and dataclasses.
- Prefer short module docstrings and plain-English inline comments only where needed.
- Keep configuration outside scripts whenever possible.

## Adding a New Experiment

1. Add a YAML file under `experiments/configs/`.
2. Reuse or extend a script in `experiments/scripts/`.
3. Document expected inputs, outputs, and random seed.
4. Store outputs under `experiments/results/` via DVC or Git LFS.

## Student Handover Checklist

Before a student branch can be merged, it must include:

- a local README in the student sandbox,
- one runnable script or importable module,
- one experiment config or protocol file,
- one short technical report or handover note,
- and a reproducibility check by the supervisor.

## Pull Requests

Every PR should include:

- a concise summary,
- impacted directories,
- whether configs changed,
- whether results need refreshing,
- and any risks or assumptions.
