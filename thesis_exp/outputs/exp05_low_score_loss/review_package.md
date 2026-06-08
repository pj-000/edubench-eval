# Exp5 L1 Review Package

Can L1 smoke start? **YES**

Can L1 formal training start? **YES**

Can L2 start? **NO until L1 reviewed**

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
| Smoke status | PASS |
| Formal training status | completed |

## Main Results

| loss | status | Accuracy | MAE_label | MAE_expected | QWK | Kendall tau | low_to_high_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L0_exp04_o3_ordinal | completed | 0.7381 | 0.3777 | 0.3430 | 0.7036 | 0.6238 | 0.2330 |
| L1_weighted_ordinal | completed | 0.7250 | 0.3894 | 0.3504 | 0.7132 | 0.6149 | 0.2136 |

## Remaining Blockers

- none for L1; formal training completed and setup/output/readability checks passed
- L2/L3/L4 remain intentionally out of scope until L1 is reviewed

## Files

- `thesis_exp/outputs/exp05_low_score_loss/report.md`
- `thesis_exp/outputs/exp05_low_score_loss/sanity_check_exp05_setup.md`
- `thesis_exp/outputs/exp05_low_score_loss/readability_check_exp05.md`
- `thesis_exp/outputs/exp05_low_score_loss/tables/loss_ablation_summary.csv`
