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

## Recommended External Layout

Keep raw public data outside the repository, for example:

- `~/datasets/alopecia_public/raw/`
- `~/datasets/alopecia_public/processed/`
- `~/datasets/alopecia_public/metadata/`
- `~/datasets/alopecia_public/manifests/`

The `exp01` manifest builder expects raw datasets to sit under
`raw/<dataset_registry_key>/`.

## Governance Rules

- Record licences, dataset URLs, and access conditions before use.
- Keep raw partner data outside this repository.
- Save audit tables, metadata summaries, and synthetic outputs only when governance permits.
