# Exp25 Structured SRC-DPO Dev Evaluation

Exp25 tests same-schema reason/score consistency preferences after Exp24 score-channel ORC-DPO
failed to move hidden low-score predictions.

## Dev Metrics

| run | parse | MAE | QWK | low-to-high | label2 recall | label5 recall | D1 pred>=4 | D1 decrease vs DPO0 | invalid D1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `exp25_r4_field_b1_ftx0_mixed_r2c` | 0.9973 | 0.4692 | 0.3181 | 0.8070 | 0.0263 | 0.8957 | 1.0000 | 0 | 0 |
| `exp25_r4_field_b3_ftx0_mixed_r2c` | 0.9955 | 0.4637 | 0.3711 | 0.7018 | 0.0263 | 0.8885 | 1.0000 | 0 | 0 |

## Structured Behavior

| run | D1 failure nonempty | D1 score_cap nonnull | low score_cap nonnull | no_major_failure on low |
|---|---:|---:|---:|---:|
| `exp25_r4_field_b1_ftx0_mixed_r2c` | 0.0000 | 0.0000 | 0.1579 | 0.4211 |
| `exp25_r4_field_b3_ftx0_mixed_r2c` | 0.0000 | 0.0000 | 0.2281 | 0.4386 |

## Transition Diagnosis

- `exp25_r4_field_b1_ftx0_mixed_r2c` vs `r2c_clean_reason_score_balanced`: changed=144 (0.1304), low_fixed=0, low_worsened=13, high_added=2.
- `exp25_r4_field_b1_ftx0_mixed_r2c` vs `r4b_shuffled_reason_balanced`: changed=219 (0.1984), low_fixed=0, low_worsened=19, high_added=2.
- `exp25_r4_field_b1_ftx0_mixed_r2c` vs `r7d_reason_real_s100_b0p03_lr5em6`: changed=259 (0.2355), low_fixed=3, low_worsened=11, high_added=2.
- `exp25_r4_field_b1_ftx0_mixed_r2c` vs `r7f_score_reason_consistency_s100_b0p03_lr5em6`: changed=151 (0.1368), low_fixed=0, low_worsened=13, high_added=2.
- `exp25_r4_field_b1_ftx0_mixed_r2c` vs `exp24_dpo0_r2c`: changed=167 (0.1513), low_fixed=5, low_worsened=0, high_added=1.
- `exp25_r4_field_b1_ftx0_mixed_r2c` vs `exp24_orc_b_r2c`: changed=155 (0.1407), low_fixed=6, low_worsened=0, high_added=1.
- `exp25_r4_field_b3_ftx0_mixed_r2c` vs `r2c_clean_reason_score_balanced`: changed=150 (0.1361), low_fixed=0, low_worsened=8, high_added=5.
- `exp25_r4_field_b3_ftx0_mixed_r2c` vs `r4b_shuffled_reason_balanced`: changed=229 (0.2078), low_fixed=0, low_worsened=13, high_added=5.
- `exp25_r4_field_b3_ftx0_mixed_r2c` vs `r7d_reason_real_s100_b0p03_lr5em6`: changed=273 (0.2486), low_fixed=7, low_worsened=11, high_added=4.
- `exp25_r4_field_b3_ftx0_mixed_r2c` vs `r7f_score_reason_consistency_s100_b0p03_lr5em6`: changed=155 (0.1407), low_fixed=0, low_worsened=8, high_added=5.
- `exp25_r4_field_b3_ftx0_mixed_r2c` vs `exp24_dpo0_r2c`: changed=171 (0.1552), low_fixed=9, low_worsened=0, high_added=3.
- `exp25_r4_field_b3_ftx0_mixed_r2c` vs `exp24_orc_b_r2c`: changed=159 (0.1445), low_fixed=10, low_worsened=0, high_added=3.

## Training Summary

| run | completed | steps | beta | pref_ftx | mean weight | peak MB | last loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `exp25_r4_field_b1_ftx0_mixed_r2c` | True | 200/200 | 1.0 | 0.0 | 0.7912 | 18395.8457 | 0.0015 |
| `exp25_r4_field_b3_ftx0_mixed_r2c` | True | 200/200 | 3.0 | 0.0 | 0.7912 | 18395.8457 | 0.0001 |

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
