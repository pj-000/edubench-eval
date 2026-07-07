# Exp25 Structured SRC-DPO Dev Evaluation

Exp25 tests same-schema reason/score consistency preferences after Exp24 score-channel ORC-DPO
failed to move hidden low-score predictions.

## Dev Metrics

| run | parse | MAE | QWK | low-to-high | label2 recall | label5 recall | D1 pred>=4 | D1 decrease vs DPO0 | invalid D1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `exp25_src_score_mismatch_r2c` | 0.9946 | 0.4905 | 0.3294 | 0.7544 | 0.0526 | 0.8471 | 1.0000 | 1 | 0 |
| `exp25_src_mixed_r2c` | 0.9946 | 0.4877 | 0.3459 | 0.7368 | 0.0526 | 0.8471 | 1.0000 | 1 | 0 |

## Structured Behavior

| run | D1 failure nonempty | D1 score_cap nonnull | low score_cap nonnull | no_major_failure on low |
|---|---:|---:|---:|---:|
| `exp25_src_score_mismatch_r2c` | 0.0000 | 0.0000 | 0.2281 | 0.3684 |
| `exp25_src_mixed_r2c` | 0.0000 | 0.0000 | 0.2456 | 0.3860 |

## Transition Diagnosis

- `exp25_src_score_mismatch_r2c` vs `r2c_clean_reason_score_balanced`: changed=128 (0.1163), low_fixed=0, low_worsened=9, high_added=12.
- `exp25_src_score_mismatch_r2c` vs `r4b_shuffled_reason_balanced`: changed=219 (0.1989), low_fixed=0, low_worsened=16, high_added=12.
- `exp25_src_score_mismatch_r2c` vs `r7d_reason_real_s100_b0p03_lr5em6`: changed=267 (0.2434), low_fixed=5, low_worsened=10, high_added=10.
- `exp25_src_score_mismatch_r2c` vs `r7f_score_reason_consistency_s100_b0p03_lr5em6`: changed=141 (0.1281), low_fixed=0, low_worsened=9, high_added=12.
- `exp25_src_score_mismatch_r2c` vs `exp24_dpo0_r2c`: changed=169 (0.1535), low_fixed=9, low_worsened=1, high_added=11.
- `exp25_src_score_mismatch_r2c` vs `exp24_orc_b_r2c`: changed=164 (0.1492), low_fixed=10, low_worsened=1, high_added=11.
- `exp25_src_mixed_r2c` vs `r2c_clean_reason_score_balanced`: changed=121 (0.1099), low_fixed=1, low_worsened=9, high_added=14.
- `exp25_src_mixed_r2c` vs `r4b_shuffled_reason_balanced`: changed=210 (0.1907), low_fixed=1, low_worsened=16, high_added=14.
- `exp25_src_mixed_r2c` vs `r7d_reason_real_s100_b0p03_lr5em6`: changed=265 (0.2416), low_fixed=6, low_worsened=9, high_added=10.
- `exp25_src_mixed_r2c` vs `r7f_score_reason_consistency_s100_b0p03_lr5em6`: changed=133 (0.1208), low_fixed=1, low_worsened=9, high_added=14.
- `exp25_src_mixed_r2c` vs `exp24_dpo0_r2c`: changed=173 (0.1571), low_fixed=10, low_worsened=0, high_added=14.
- `exp25_src_mixed_r2c` vs `exp24_orc_b_r2c`: changed=166 (0.1510), low_fixed=11, low_worsened=0, high_added=14.

## Training Summary

| run | completed | steps | beta | pref_ftx | mean weight | peak MB | last loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `exp25_src_score_mismatch_r2c` | True | 100/100 | 0.03 | 0.05 | 1.0000 | 18362.0688 | 0.7725 |
| `exp25_src_mixed_r2c` | True | 100/100 | 0.03 | 0.05 | 0.7912 | 18395.8457 | 0.8623 |

## Decision

- recommendation: `src_dpo_not_yet_successful_consider_hidden_failure_expansion`
- reason: No Exp25 SRC-DPO run satisfies minimum success.
- minimum_success_runs: none
- strong_success_runs: none
- missing predictions: none

## Guardrails

- No test split is read in this collector.
- Dev labels are used only for evaluation.
- Human rationale is not included in prediction prompts.
- Raw predictions, logs, checkpoints, and adapter weights must not be committed.
