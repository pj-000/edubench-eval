# Exp5 Review Package

Can L1 smoke start? **YES**

Can L1 formal training start? **YES**

Can L2 smoke start? **YES**

Can L2 formal training start? **YES**

Can L3b smoke start? **YES**

Can L3b formal training start? **NO**

Can L3a or L4 start? **NO, intentionally not implemented**

Can Exp6 start? **NO until L3b is reviewed**

## Class Weights Summary

| label_5 | train_count | raw_weight | clipped_weight |
| ---: | ---: | ---: | ---: |
| 1 | 24 | 22.1167 | 3.0000 |
| 2 | 52 | 10.2077 | 3.0000 |
| 3 | 251 | 2.1147 | 2.1147 |
| 4 | 946 | 0.5611 | 0.5611 |
| 5 | 1381 | 0.3844 | 0.5000 |

## Status

| item | status |
| --- | --- |
| Setup sanity status | PASS |
| Readability status | PASS |
| L2 toy loss check status | PASS |
| L3b toy loss check status | PASS |
| L1 smoke status | PASS |
| L1 formal status | completed |
| L2 smoke status | PASS |
| L2 formal status | completed |
| L3b smoke status | PENDING |
| L3b formal status | completed |

## Main Results

| loss | status | Accuracy | MAE_label | MAE_expected | QWK | Kendall tau | low_to_high_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L0_exp04_o3_ordinal | completed | 0.7381 | 0.3777 | 0.3430 | 0.7036 | 0.6238 | 0.2330 |
| L1_weighted_ordinal | completed | 0.7250 | 0.3894 | 0.3504 | 0.7132 | 0.6149 | 0.2136 |
| L2a_asymmetric_ordinal_lambda03_margin0 | completed | 0.7160 | 0.4080 | 0.3705 | 0.6356 | 0.5991 | 0.3398 |
| L2b_asymmetric_ordinal_lambda05_margin0 | completed | 0.7164 | 0.3971 | 0.3633 | 0.6745 | 0.6038 | 0.2718 |
| L3b_weighted_threshold_mu03 | completed | 0.7205 | 0.3946 | 0.3602 | 0.6871 | 0.6122 | 0.2330 |

## Remaining Blockers

- none for L3b; formal training completed and setup/output/readability checks passed
- L3a/L4 remain intentionally out of scope
- Exp6 should wait until L3b results are reviewed

## Files

- `thesis_exp/outputs/exp05_low_score_loss/report.md`
- `thesis_exp/outputs/exp05_low_score_loss/sanity_check_exp05_setup.md`
- `thesis_exp/outputs/exp05_low_score_loss/readability_check_exp05.md`
- `thesis_exp/outputs/exp05_low_score_loss/tables/loss_ablation_summary.csv`
- `thesis_exp/outputs/exp05_low_score_loss/tables/l3b_toy_loss_checks.csv`
