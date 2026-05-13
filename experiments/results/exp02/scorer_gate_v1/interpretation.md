# Interpretation

The Experiment 02 scorer entry gate passed.

The exact seven-class model reached macro F1 `0.424` over all train-observed
labels and `0.593` over validation-present labels, clearing the pre-specified
`0.40` gate. The ordinal three-bin formulation performed better, with macro F1
`0.552` over all train-observed bins and `0.829` over validation-present bins.

The validation set contains severities `3`, `4`, `5`, `6`, and `7`, with
severities `1` and `2` absent. The `0.593` validation-present macro F1 therefore
summarizes performance only over classes present in validation. The stricter
all-train-label macro F1 remains above gate (`0.424 >= 0.40`), which is the
reason the scorer is accepted as an Experiment 02 engineering gate.

Feature-based classifiers were used because the Experiment 01 neural scorer was
too weak for the entry gate on the current small dataset, and this run needed an
auditable small-data evaluator. This does not retire the neural scorer path; it
postpones it until there is enough data, augmentation, or pseudo-paired
supervision to train a neural evaluator without simply overfitting.

This is a meaningful improvement over the Experiment 01 neural severity scorer,
which had validation macro F1 `0.15`. However, the result is still fragile
because validation contains only `9` samples and does not include severity
classes `1` or `2`.

The scorer is therefore acceptable as an Experiment 02 engineering gate and
proxy-QA signal. It is not strong enough to serve as a clinical endpoint or as
the only evaluation for learned prognosis generation.
