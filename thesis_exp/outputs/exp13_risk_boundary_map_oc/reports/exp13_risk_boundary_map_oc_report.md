# Exp13 Risk-Boundary MAP-OC

Mode: `formal`
Status: `COMPLETED`

Scout ranking is dev-only. Test metrics are not used for checkpoint selection, config selection, or
tuning.

## Dev-Only Selected Rows

| run | seed | epoch | dev MAE | dev low-to-high | dev p_gt_3_low_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| score_proj_l2h_lam0p20 | 43 | 1 | 0.4899 | 0.4912 | 0.4862 |
| score_proj_l2h_lam0p20 | 42 | 2 | 0.4827 | 0.4912 | 0.4896 |
| score_proj_l2h_lam0p20 | 44 | 3 | 0.4878 | 0.4912 | 0.4934 |

## Method Notes

- Exp13 adds a squared hinge risk-boundary loss on projected `q3 = P(y > 3)` for low-score samples.
- The base setting is Exp12B `train_projection_point_pair` unless a config explicitly switches base.
- PAVA projection preserves ordinal threshold positions; it is not sorting.
- This is a next-stage experiment design for low-to-high risk calibration, not evidence that the problem is solved.
