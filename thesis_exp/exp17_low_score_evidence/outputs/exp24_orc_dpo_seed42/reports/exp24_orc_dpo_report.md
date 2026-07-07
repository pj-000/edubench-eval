# Exp24 Score-Channel ORC-DPO Dev Evaluation

Exp24 uses score-only DPO responses plus ordinal/risk weights and margins.
Human rationales are auxiliary targets, not contrasted directly against rejected score responses.

## Dev Metrics

| run | dataset | parse | MAE | QWK | low-to-high | high-to-low | label2 recall | label5 recall | D1 pred>=4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `r2c_clean_reason_score_balanced` | `none_init_baseline` | 1.0000 | 0.4219 | 0.4982 | 0.5965 | 0.0000 | 0.0000 | 0.8435 | 1.0000 |
| `r4b_shuffled_reason_balanced` | `none_init_baseline` | 1.0000 | 0.4074 | 0.5617 | 0.4737 | 0.0063 | 0.0526 | 0.8363 | 0.8462 |
| `r7d_reason_real_s100_b0p03_lr5em6` | `human_reason_real_error` | 0.9964 | 0.5739 | 0.3258 | 0.6842 | 0.0455 | 0.2632 | 0.7986 | 0.6923 |
| `r7e_matched_score_only_s100_b0p03_lr5em6` | `matched_score_only_control` | 1.0000 | 0.4959 | 0.3833 | 0.8596 | 0.0095 | 0.0789 | 0.5809 | 0.9615 |
| `r7f_score_reason_consistency_s100_b0p03_lr5em6` | `score_reason_consistency_counterfactual` | 1.0000 | 0.4155 | 0.5040 | 0.5965 | 0.0000 | 0.0000 | 0.8399 | 1.0000 |

## Training Summary

| run | completed | steps | alpha_lh | alpha_hl | alpha_d | margin_lh | margin_hl | margin_d | lambda_reason | init delta | mean weight | mean margin | peak MB | last loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `exp24_dpo0_r2c` | False | / |  |  |  |  |  |  |  | nan | nan | nan | nan | nan |
| `exp24_orc_a_r2c` | False | / |  |  |  |  |  |  |  | nan | nan | nan | nan | nan |
| `exp24_orc_b_r2c` | False | / |  |  |  |  |  |  |  | nan | nan | nan | nan | nan |
| `exp24_orc_b_noreason_r2c` | False | / |  |  |  |  |  |  |  | nan | nan | nan | nan | nan |
| `exp24_orc_c_r2c` | False | / |  |  |  |  |  |  |  | nan | nan | nan | nan | nan |

## Structured Failure Diagnostics

| run | failure micro-F1 | D1 nonempty failure | score_cap nonnull |
|---|---:|---:|---:|
| `r2c_clean_reason_score_balanced` | 0.0000 | 0.0000 | nan |
| `r4b_shuffled_reason_balanced` | 0.0000 | 0.0000 | nan |
| `r7d_reason_real_s100_b0p03_lr5em6` | 0.0000 | 0.0000 | 0.2692 |
| `r7e_matched_score_only_s100_b0p03_lr5em6` | 0.0000 | 0.0000 | 0.0385 |
| `r7f_score_reason_consistency_s100_b0p03_lr5em6` | 0.0000 | 0.0000 | 0.0000 |

## Decision

- recommendation: `wait_for_exp24_predictions`
- reason: No Exp24 ORC predictions were collected.
- best_by_low_to_high_mae_qwk: ``
- minimum_success_runs: none
- strong_success_runs: none

## Sources

- included Exp23/baseline rows: r2c_clean_reason_score_balanced, r4b_shuffled_reason_balanced, r7d_reason_real_s100_b0p03_lr5em6, r7e_matched_score_only_s100_b0p03_lr5em6, r7f_score_reason_consistency_s100_b0p03_lr5em6
- missing Exp23/baseline rows: none
- missing Exp24 predictions: exp24_dpo0_r2c, exp24_orc_a_r2c, exp24_orc_b_r2c, exp24_orc_b_noreason_r2c, exp24_orc_c_r2c

## Guardrails

- Test split is read only for sample_id/question-key leakage guardrails in data preparation.
- Test labels are not read or used for training, selection, tuning, or evaluation.
- Dev labels are used only for evaluation.
- Human rationale is not included in the prediction prompt.
- Raw predictions, logs, checkpoints, and adapter weights must not be committed.
