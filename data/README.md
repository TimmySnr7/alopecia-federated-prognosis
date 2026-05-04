# Data Layer

This repository does not store raw clinical data in Git.

## Intended Dataset Families

- public alopecia severity datasets,
- broader dermatology resources for transfer or auditing,
- fairness-focused datasets such as Fitzpatrick17k or DDI,
- and later site-local partner datasets accessed under approved governance.

## Registry

The canonical public-dataset registry for the first proof-of-concept lives in
`data/dataset_registry.yaml`. Every experiment config should refer to datasets
by registry key rather than by informal name only.

## Governance Rules

- Record licences, dataset URLs, and access conditions before use.
- Keep raw partner data outside this repository.
- Save audit tables, metadata summaries, and synthetic outputs only when governance permits.
