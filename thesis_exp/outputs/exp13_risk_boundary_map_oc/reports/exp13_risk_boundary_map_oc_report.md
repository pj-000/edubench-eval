# Exp13 Risk-Boundary MAP-OC

Mode: `scout`
Status: `COMPLETED`

Scout ranking is dev-only. Test metrics are not used for checkpoint selection, config selection, or
tuning.

## Dev-Only Selected Rows

| run | seed | epoch | dev MAE | dev low-to-high | dev p_gt_3_low_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| map_oc_full_l2h_lam0p20 | 42 | 2 | 0.4914 | 0.4912 | 0.4866 |
| score_proj_l2h_lam0p20 | 42 | 2 | 0.4827 | 0.4912 | 0.4896 |
| point_pair_proj_l2h_lam0p20 | 42 | 2 | 0.4893 | 0.4912 | 0.4905 |
| score_proj_t3_brier_lam0p05 | 42 | 3 | 0.4833 | 0.4912 | 0.4922 |
| point_pair_proj_l2h_label2_w1p5_lam0p20 | 42 | 2 | 0.4863 | 0.4912 | 0.4923 |
| point_pair_proj_l2h_lam0p40 | 42 | 2 | 0.4866 | 0.4912 | 0.4932 |
| point_pair_proj_l2h_lam0p20_no_mono | 42 | 1 | 0.4908 | 0.4912 | 0.4938 |
| point_pair_proj_l2h_lam0p10 | 42 | 2 | 0.4899 | 0.4912 | 0.4952 |

## Method Notes

- Exp13 adds a squared hinge risk-boundary loss on projected `q3 = P(y > 3)` for low-score samples.
- The base setting is Exp12B `train_projection_point_pair` unless a config explicitly switches base.
- PAVA projection preserves ordinal threshold positions; it is not sorting.
- This is a next-stage experiment design for low-to-high risk calibration, not evidence that the problem is solved.
