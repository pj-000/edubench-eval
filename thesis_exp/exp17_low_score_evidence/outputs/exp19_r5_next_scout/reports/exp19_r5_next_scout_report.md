# Exp19-R5 Next Scout Dev Evaluation

This report evaluates the R5E-from-R1b control and R5C-no-mid score-risk scouts on the original dev
split.
Raw predictions/logs/checkpoints remain gitignored. No test split is read.

## Dev Metrics

| run | init | dataset | MAE | QWK | low-to-high | high-to-low | label2 recall |
|---|---|---|---:|---:|---:|---:|---:|
| `r5e_from_r1b` | `r1b_score_only_balanced` | `r5e_hard_synthetic_control` | 0.3939 | 0.5606 | 0.5088 | 0.0021 | 0.0000 |
| `r5c_no_mid_from_r1b` | `r1b_score_only_balanced` | `r5c_score_risk_no_mid` | 0.3893 | 0.5705 | 0.4912 | 0.0021 | 0.0263 |
| `r5c_no_mid_from_r2c` | `r2c_clean_reason_score_balanced` | `r5c_score_risk_no_mid` | 0.4724 | 0.4551 | 0.5614 | 0.0138 | 0.0526 |
| `r1b_score_only_balanced` | `r1b_score_only_balanced` | `none_init_baseline` | 0.3975 | 0.5565 | 0.5263 | 0.0021 | 0.0000 |
| `r2c_clean_reason_score_balanced` | `r2c_clean_reason_score_balanced` | `none_init_baseline` | 0.4219 | 0.4982 | 0.5965 | 0.0000 | 0.0000 |
| `r4b_shuffled_reason_balanced` | `r4b_shuffled_reason_balanced` | `none_init_baseline` | 0.4074 | 0.5617 | 0.4737 | 0.0063 | 0.0526 |
| `r5c_from_r1b` | `r1b_score_only_balanced` | `r5c_score_risk` | 0.3866 | 0.5714 | 0.4912 | 0.0021 | 0.0263 |
| `r5c_from_r2c` | `r2c_clean_reason_score_balanced` | `r5c_score_risk` | 0.4544 | 0.4860 | 0.5263 | 0.0138 | 0.0263 |
| `r5e_from_r2c` | `r2c_clean_reason_score_balanced` | `r5e_hard_synthetic_control` | 0.4535 | 0.4921 | 0.5439 | 0.0085 | 0.0263 |

## D1 Hidden And Failure-Type Tables

| run | D1 pred>=4 | D1 label2 recall | failure micro-F1 | D1 nonempty failure | score_cap nonnull |
|---|---:|---:|---:|---:|---:|
| `r5e_from_r1b` | 0.9615 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `r5c_no_mid_from_r1b` | 0.9231 | 0.0385 | 0.0000 | 0.0000 | 0.0000 |
| `r5c_no_mid_from_r2c` | 0.9615 | 0.0385 | 0.0000 | 0.0385 | 0.0385 |
| `r1b_score_only_balanced` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | nan |
| `r2c_clean_reason_score_balanced` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | nan |
| `r4b_shuffled_reason_balanced` | 0.8462 | 0.0385 | 0.0000 | 0.0000 | nan |
| `r5c_from_r1b` | 0.9231 | 0.0385 | 0.0000 | 0.0000 | nan |
| `r5c_from_r2c` | 0.9615 | 0.0385 | 0.0000 | 0.0385 | nan |
| `r5e_from_r2c` | 0.9615 | 0.0385 | 0.0000 | 0.0385 | nan |

## Required Questions

- R5E from R1b similar to R5C from R1b: `True` (low-to-high gap=0.0175).
- R5C no-mid from R1b low-to-high improvement over R5C main: -0.0000.
- R5C no-mid from R2c low-to-high improvement over R5C main: -0.0351.
- recommendation: `prioritize_r5f_rejection_mining`.
- reason: R5E from R1b is close to R5C from R1b; synthetic-control effects remain a concern.

## Sources

- included prior rows: r1b_score_only_balanced, r2c_clean_reason_score_balanced, r4b_shuffled_reason_balanced, r5c_from_r1b, r5c_from_r2c, r5e_from_r2c
- missing prior rows: none
- missing new prediction runs: none

## Guardrails

- Evaluation uses the original dev split, not balanced train distribution.
- Test split is not read.
- D1 annotations are evaluation references only.
- Human rationale is not included in the prediction prompt.
