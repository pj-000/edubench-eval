# Exp4 Review Package

Can server smoke test start? **YES**

Can formal O2/O3 training start? **COMPLETED**

## Setup Status

| item | status | notes |
| --- | --- | --- |
| O1 reuse status | PASS | run status: reused_exp03_a4 |
| O2 regression setup status | PASS | uses `human_mean_5`, SmoothL1Loss |
| O3 ordinal setup status | PASS | uses four BCE threshold labels |
| readability status | PASS | `readability_check_exp04` |
| setup sanity status | PASS | `sanity_check_exp04_setup` |

## Main Results

| objective | status | Accuracy | MAE_label | MAE_expected | QWK | Kendall tau | severe_error_rate | low_to_high_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O1_classification | reused_exp03_a4 | 0.7412 | 0.4031 | 0.3666 | 0.6233 | 0.5941 | 0.0428 | 0.4466 |
| O2_regression_smoothl1 | completed | 0.4820 | 0.6162 | 0.5616 | 0.2390 | 0.3652 | 0.0528 | 0.9709 |
| O3_ordinal | completed | 0.5289 | 0.6121 | 0.5646 | 0.0719 | 0.2325 | 0.0586 | 1.0000 |

## Remaining Blockers

- none; formal O2/O3 completed

## Files

- `thesis_exp/outputs/exp04_target_objectives/report.md`
- `thesis_exp/outputs/exp04_target_objectives/sanity_check_exp04_setup.md`
- `thesis_exp/outputs/exp04_target_objectives/readability_check_exp04.md`
- `thesis_exp/outputs/exp04_target_objectives/tables/target_objective_summary.csv`
