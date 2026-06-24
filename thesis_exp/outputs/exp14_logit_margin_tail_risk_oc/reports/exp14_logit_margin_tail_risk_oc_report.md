# Exp14 Logit-Margin Tail-Risk OC

Mode: `scout`
Status: `COMPLETED`
Formal recommendation: `NO_FORMAL_RECOMMENDED`

Scout ranking is dev-only. Test metrics are not used for checkpoint selection, config selection, or
tuning.

Exp13 formal dev baseline: dev_low_to_high_count = 28/57; best dev_MAE around 0.4827.

A config is recommended for formal only if dev_low_to_high_count <= 26/57, dev_MAE <= 0.493,
high-to-low does not clearly increase, and label4/label5 recall do not collapse.

## Dev-Only Selected Rows

| run | epoch | dev MAE | dev low-to-high count | dev p_gt_3_low_mean | dev low_z3_q95 | formal candidate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| point_pair_tail_logit_margin_lam0p02_top0p50 | 2 | 0.4824 | 28/57 | 0.4914 | 7.0104 | False |
| score_logit_margin_lam0p01_alllow | 2 | 0.4869 | 28/57 | 0.4910 | 7.2611 | False |
| score_tail_logit_margin_lam0p05_top0p50 | 2 | 0.4875 | 28/57 | 0.4928 | 7.2712 | False |
| score_tail_logit_margin_lam0p02_top0p50 | 2 | 0.4878 | 28/57 | 0.4899 | 6.9697 | False |
| score_tail_logit_margin_lam0p02_top0p25 | 3 | 0.4884 | 28/57 | 0.4935 | 7.3782 | False |
| score_logit_margin_lam0p02_alllow | 3 | 0.4887 | 28/57 | 0.4944 | 6.7988 | False |
| score_logit_margin_lam0p05_alllow | 2 | 0.4896 | 28/57 | 0.4930 | 6.7944 | False |

## Method Notes

- Exp14 adds a squared hinge tail-risk loss directly on raw threshold-3 logit `z3` for `y <= 2` samples.
- Decode still uses projected probabilities; the loss does not use hard decoded labels.
- PAVA projection is reused for decoding and diagnostics; this experiment does not replace it with sorting.
- This is a dev-only scout for the saturated threshold-3 boundary exposed by Exp13, not evidence that the problem is solved.
