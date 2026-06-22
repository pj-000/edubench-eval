# Exp11 Checkpoint Selection Sensitivity

Completed seeds: `3`

This report uses dev-only checkpoint selection rules. Test metrics are post-hoc diagnostic only and
are not used for selection rule tuning or training.

## Selection Answers

| question | selected epoch | dev MAE | dev low-to-high | test low-to-high | test MAE | test QWK |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev_mae_min | 3 | 0.4851 | 0.4912 | 0.3871 | 0.4192 | 0.6084 |
| dev_qwk_max | 3 | 0.4851 | 0.4912 | 0.3871 | 0.4192 | 0.6084 |
| dev_low_to_high_min_diagnostic | 3 | 0.4851 | 0.4912 | 0.3871 | 0.4192 | 0.6084 |
| mae_guard_soft_risk | 3 | 0.4851 | 0.4912 | 0.3871 | 0.4192 | 0.6084 |
| mae_guard_label2_soft_risk | 3 | 0.4851 | 0.4912 | 0.3871 | 0.4192 | 0.6084 |
| mae_guard_p_gt_3_low_mean | 3 | 0.4851 | 0.4912 | 0.3871 | 0.4192 | 0.6084 |
| mae_guard_soft_risk_mono | 3 | 0.4851 | 0.4912 | 0.3871 | 0.4192 | 0.6084 |
| last_epoch_diagnostic | 3 | 0.4851 | 0.4912 | 0.3871 | 0.4192 | 0.6084 |

## Findings

- Different selection rules do not change post-hoc test low-to-high under the first completed seed.
- Dev hard low-to-high unique values: `['0.4912']`. If this list has one value, hard count has little selection signal under this seed.
- Dev soft-risk values among selected rules: `['0.5037']`. Soft risk is useful when it varies across epochs inside the MAE guard.
- Best post-hoc delta versus dev_mae_min: rule `mae_guard_soft_risk` with delta test low-to-high count `-1`.
- If risk-aware selection does not reduce post-hoc test low-to-high, treat that as a negative result rather than evidence of improvement.
- The low-score test subset is small, so count-level interpretation is necessary.

## Selection Rules

- `dev_mae_min`
- `dev_qwk_max`
- `dev_low_to_high_min_diagnostic`
- `mae_guard_soft_risk`
- `mae_guard_label2_soft_risk`
- `mae_guard_p_gt_3_low_mean`
- `mae_guard_soft_risk_mono`
- `last_epoch_diagnostic`

## Soft Risk Definition

For true low-score samples (`y <= 2`), `soft_low_to_high = mean(sigmoid(gamma * (r_theta(x) -
3.5)))`, where `r_theta(x) = 1 + sum_t p_t(x)` and `gamma=4.0` by default.
`p_gt_3_low_mean` is the low-score mean of `P(y > 3 | x)`, using the existing `prob_gt_3` prediction
column.
