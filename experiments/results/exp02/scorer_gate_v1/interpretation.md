# Interpretation

The Experiment 02 scorer entry gate passed.

The exact seven-class model reached macro F1 `0.424` over all train-observed
labels and `0.593` over validation-present labels, clearing the pre-specified
`0.40` gate. The ordinal three-bin formulation performed better, with macro F1
`0.552` over all train-observed bins and `0.829` over validation-present bins.

This is a meaningful improvement over the Experiment 01 neural severity scorer,
which had validation macro F1 `0.15`. However, the result is still fragile
because validation contains only `9` samples and does not include source classes
`1` or `2`.

The scorer is therefore acceptable as an Experiment 02 engineering gate and
proxy-QA signal. It is not strong enough to serve as a clinical endpoint or as
the only evaluation for learned prognosis generation.
