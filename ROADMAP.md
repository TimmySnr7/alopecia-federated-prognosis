# Roadmap

## Purpose

This roadmap keeps the repository useful in two scenarios:

- the PhD progresses entirely through the core workspace, and
- students contribute scoped artefacts that are later promoted into the main codebase.

## Phase 1: Foundation

- finalise public-data audit pipeline
- establish severity-grading baseline
- stabilise experiment config conventions
- keep evaluation metrics importable and tested

## Phase 2: Early Proof of Concept

- run centralised latent-diffusion baseline
- complete conditioning ablations
- generate first reproducible demo outputs
- draft early workshop or preprint material

## Phase 3: Simulated Federation

- implement client partitioning and federation baselines
- compare FedAvg and FedProx
- run DP sweeps and capture privacy-utility-calibration curves
- generate fairness and robustness summaries

## Phase 4: Human Evaluation and Governance

- operationalise the 2AFC realism protocol
- run the formative walkthrough study
- refine GDPR and POPIA governance artefacts
- promote reusable outputs into `docs/` and `thesis_outputs/`

## Phase 5: Partner-Ready Research Platform

- prepare for site-local datasets and federated deployment
- integrate approved clinical workflows without moving raw data into Git
- align experiment reports with thesis chapters and publications

## Promotion Rule

Student work should only move from `student_projects/` into the core repository when it is:

- reproducible,
- documented,
- reviewed,
- and not a hidden dependency of any unfinished student branch.
