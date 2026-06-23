# Exp14 Logit-Margin Tail-Risk OC

Mode: `scout`
Status: `NO_COMPLETED_RUNS`
Formal recommendation: `NO_FORMAL_RECOMMENDED`

Scout ranking is dev-only. Test metrics are not used for checkpoint selection, config selection, or
tuning.

Exp13 formal dev baseline: dev_low_to_high_count = 28/57; best dev_MAE around 0.4827.

A config is recommended for formal only if dev_low_to_high_count <= 26/57, dev_MAE <= 0.493,
high-to-low does not clearly increase, and label4/label5 recall do not collapse.

## Dev-Only Selected Rows

No completed Exp14 scout rows were found under the local ignored runs directory.

## Method Notes

- Exp14 adds a squared hinge tail-risk loss directly on raw threshold-3 logit `z3` for `y <= 2` samples.
- Decode still uses projected probabilities; the loss does not use hard decoded labels.
- PAVA projection is reused for decoding and diagnostics; this experiment does not replace it with sorting.
- This is a dev-only scout for the saturated threshold-3 boundary exposed by Exp13, not evidence that the problem is solved.
