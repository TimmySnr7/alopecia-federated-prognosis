# Human Face-Validity Protocol

## Purpose

Assess whether the fragility score aligns with human face-validity ratings of alopecia severity-sweep outputs.

## Pipelines

Use only:

- `P1_proxy`
- `P2_diffusion_archived_exp06`
- `P4_sparse4_ridge`

`P3_sparse2_ridge` is excluded from rater review to reduce burden; it remains the computational production candidate.

## Raters

Recruit 2-4 raters: dermatologists, dermatology registrars, or trained clinical image assessors.

## Rating Task

For each displayed severity sweep, raters score five dimensions on a 1-5 Likert scale:

| Dimension | Question |
|---|---|
| Source preservation | Does the output preserve the original source identity/structure? |
| Scalp localisation | Are edits restricted to the scalp/hair region? |
| Monotonic progression | Does the sweep show coherent severity increase? |
| Anatomical plausibility | Are the outputs anatomically believable? |
| Overall acceptability | Would this be acceptable as a research visualisation? |

## Analysis

Compute mean acceptability per pipeline and correlate overall acceptability with balanced fragility score using Spearman correlation. Expected direction:

`fragility score increases -> human acceptability decreases`

The prepared data-entry template is `human_face_validity_rating_template.csv`.
