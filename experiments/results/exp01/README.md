# Experiment 1 Results

This directory stores compact, reproducible result artefacts for the early
centralised baseline work in Experiment 1.

Result packages should prefer:

- machine-readable summaries such as `summary.json`,
- short interpretation notes,
- lightweight tables or plots that can be regenerated from saved summaries,
- and references back to the exact manifest files used.

Large binaries, model checkpoints, or image-heavy outputs should be tracked with
DVC or Git LFS instead of plain Git.
- `generative_scaffold_working_v1/`: first successful centralised generative scaffold result for the formal working subset.
- `generative_scaffold_working_v8/`: first external frozen-scorer guidance result, recorded as a diagnostic milestone before the adjacent-shift `v9` refinement.
- `generative_scaffold_working_v9/`: adjacent-shift frozen-guidance result showing better optimization but visually collapsed sweeps, motivating the rebalanced `v10` loss.
- `generative_scaffold_working_v10/`: rebalanced guidance attempt that still collapsed visually, motivating the residual-prediction `v11` pivot.
