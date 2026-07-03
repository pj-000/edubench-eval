# Exp19-R5F2 DPO Scout Dev Evaluation

This report evaluates the R5F2 expanded rejection-mined DPO scout on the original question-disjoint
dev split.
Raw predictions/logs/checkpoints stay gitignored. No test split is read.

## Dev Metrics

| run | init | dataset | MAE | QWK | low-to-high | high-to-low | label2 recall |
|---|---|---|---:|---:|---:|---:|---:|
| `r5f2_main_from_r1b` | `r1b_score_only_balanced` | `r5f2_score_risk_main` | 0.3866 | 0.5608 | 0.5088 | 0.0021 | 0.0000 |
| `r5f2_main_from_r2c` | `r2c_clean_reason_score_balanced` | `r5f2_score_risk_main` | 0.4381 | 0.5156 | 0.5263 | 0.0127 | 0.0789 |
| `r5f2_real_only_from_r1b` | `r1b_score_only_balanced` | `r5f2_real_only_small` | 0.3939 | 0.5685 | 0.4912 | 0.0021 | 0.0526 |
| `r5f2_real_only_from_r2c` | `r2c_clean_reason_score_balanced` | `r5f2_real_only_small` | 0.6748 | 0.4031 | 0.2632 | 0.0847 | 0.3158 |
| `r1b_score_only_balanced` | `r1b_score_only_balanced` | `none_init_baseline` | 0.3975 | 0.5565 | 0.5263 | 0.0021 | 0.0000 |
| `r2c_clean_reason_score_balanced` | `r2c_clean_reason_score_balanced` | `none_init_baseline` | 0.4219 | 0.4982 | 0.5965 | 0.0000 | 0.0000 |
| `r4b_shuffled_reason_balanced` | `r4b_shuffled_reason_balanced` | `none_init_baseline` | 0.4074 | 0.5617 | 0.4737 | 0.0063 | 0.0526 |
| `r5c_from_r1b` | `r1b_score_only_balanced` | `r5c_score_risk` | 0.3866 | 0.5714 | 0.4912 | 0.0021 | 0.0263 |
| `r5c_from_r2c` | `r2c_clean_reason_score_balanced` | `r5c_score_risk` | 0.4544 | 0.4860 | 0.5263 | 0.0138 | 0.0263 |
| `r5e_from_r2c` | `r2c_clean_reason_score_balanced` | `r5e_hard_synthetic_control` | 0.4535 | 0.4921 | 0.5439 | 0.0085 | 0.0263 |
| `r5e_from_r1b` | `r1b_score_only_balanced` | `r5e_hard_synthetic_control` | 0.3939 | 0.5606 | 0.5088 | 0.0021 | 0.0000 |
| `r5c_no_mid_from_r1b` | `r1b_score_only_balanced` | `r5c_score_risk_no_mid` | 0.3893 | 0.5705 | 0.4912 | 0.0021 | 0.0263 |
| `r5c_no_mid_from_r2c` | `r2c_clean_reason_score_balanced` | `r5c_score_risk_no_mid` | 0.4724 | 0.4551 | 0.5614 | 0.0138 | 0.0526 |

## D1 Hidden And Failure-Type Tables

| run | D1 pred>=4 | D1 label2 recall | failure micro-F1 | D1 nonempty failure | score_cap nonnull |
|---|---:|---:|---:|---:|---:|
| `r5f2_main_from_r1b` | 0.9615 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `r5f2_main_from_r2c` | 0.9615 | 0.0000 | 0.0000 | 0.0385 | 0.0385 |
| `r5f2_real_only_from_r1b` | 0.9231 | 0.0385 | 0.0000 | 0.0000 | 0.0000 |
| `r5f2_real_only_from_r2c` | 0.5385 | 0.2308 | 0.0000 | 0.2308 | 0.4615 |
| `r5c_from_r1b` | 0.9231 | 0.0385 | 0.0000 | 0.0000 | nan |
| `r5c_from_r2c` | 0.9615 | 0.0385 | 0.0000 | 0.0385 | nan |
| `r5e_from_r2c` | 0.9615 | 0.0385 | 0.0000 | 0.0385 | nan |
| `r5e_from_r1b` | 0.9615 | 0.0000 | 0.0000 | 0.0000 | nan |
| `r5c_no_mid_from_r1b` | 0.9231 | 0.0385 | 0.0000 | 0.0000 | nan |
| `r5c_no_mid_from_r2c` | 0.9615 | 0.0385 | 0.0000 | 0.0385 | nan |

## Required Questions

- Does R5F2 score-risk main from R1b reduce low-to-high vs R1b: 0.0175.
- Does R5F2 score-risk main from R2c reduce low-to-high vs R2c: 0.0702.
- Does R5F2 score-risk main beat hard-synthetic controls: R1b=False, R2c=False.
- Does real-only diagnostic behave differently from score-risk main: compare the real-only rows above; real-only is diagnostic and not a full-DPO candidate by itself.
- Does any run damage high-score protection: inspect high-to-low rates; the success rule allows at most +0.05.

## Decision

- recommendation: `continue_rejection_mining_or_adjust_dpo`
- reason: R5F2 gives some risk-side movement, but it does not pass the guarded full-DPO rule.

## Sources

- included prior rows: r1b_score_only_balanced, r2c_clean_reason_score_balanced, r4b_shuffled_reason_balanced, r5c_from_r1b, r5c_from_r2c, r5e_from_r2c, r5e_from_r1b, r5c_no_mid_from_r1b, r5c_no_mid_from_r2c
- missing prior rows: none
- missing new prediction runs: none

## Guardrails

- Evaluation uses the original dev split, not a balanced train distribution.
- Test split is not read.
- D1 annotations are evaluation references only.
- Human rationale is not included in the prediction prompt.
- R5F2 plus-evidence data is not trained in this scout.
