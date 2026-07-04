# Exp19-R5G Risk-Calibrated DPO Scout Dev Evaluation

R5G tests lighter real-only DPO and ratio-calibrated low/high DPO after R5F2 showed a
low-risk improvement with over-conservative side effects.

## Dev Metrics

| run | init | dataset | MAE | QWK | low-to-high | high-to-low | label2 recall | label5 recall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `r5g_a1_real_only_s25_b0p03_lr2em6` | `r2c_clean_reason_score_balanced` | `r5g_real_only` | 0.4228 | 0.5159 | 0.5614 | 0.0000 | 0.0263 | 0.8327 |
| `r5g_a2_real_only_s50_b0p03_lr2em6` | `r2c_clean_reason_score_balanced` | `r5g_real_only` | 0.4173 | 0.5410 | 0.5263 | 0.0011 | 0.0526 | 0.8273 |
| `r5g_a3_real_only_s50_b0p05_lr5em6` | `r2c_clean_reason_score_balanced` | `r5g_real_only` | 0.4905 | 0.5010 | 0.4035 | 0.0212 | 0.1579 | 0.7716 |
| `r5g_b1_ratio70_30_s100_b0p03_lr5em6` | `r2c_clean_reason_score_balanced` | `r5g_ratio_70_30` | 0.4752 | 0.4901 | 0.4211 | 0.0265 | 0.1316 | 0.7770 |
| `r5g_b2_ratio60_40_s100_b0p03_lr5em6` | `r2c_clean_reason_score_balanced` | `r5g_ratio_60_40` | 0.4435 | 0.5095 | 0.5088 | 0.0148 | 0.1053 | 0.8004 |
| `r5g_b3_ratio50_50_s100_b0p03_lr5em6` | `r2c_clean_reason_score_balanced` | `r5g_ratio_50_50` | 0.4390 | 0.5133 | 0.5263 | 0.0127 | 0.0526 | 0.8058 |
| `r2c_clean_reason_score_balanced` | `r2c_clean_reason_score_balanced` | `none_init_baseline` | 0.4219 | 0.4982 | 0.5965 | 0.0000 | 0.0000 | 0.8435 |
| `r4b_shuffled_reason_balanced` | `r4b_shuffled_reason_balanced` | `none_init_baseline` | 0.4074 | 0.5617 | 0.4737 | 0.0063 | 0.0526 | 0.8363 |
| `r5f2_main_from_r2c` | `r2c_clean_reason_score_balanced` | `r5f2_score_risk_main` | 0.4381 | 0.5156 | 0.5263 | 0.0127 | 0.0789 | 0.8058 |
| `r5f2_real_only_from_r2c` | `r2c_clean_reason_score_balanced` | `r5f2_real_only_small` | 0.6748 | 0.4031 | 0.2632 | 0.0847 | 0.3158 | 0.6619 |

## D1 Hidden And Failure-Type Tables

| run | D1 pred>=4 | D1 label2 recall | failure micro-F1 | D1 nonempty failure | score_cap nonnull |
|---|---:|---:|---:|---:|---:|
| `r5g_a1_real_only_s25_b0p03_lr2em6` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `r5g_a2_real_only_s50_b0p03_lr2em6` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `r5g_a3_real_only_s50_b0p05_lr5em6` | 0.8077 | 0.0769 | 0.0000 | 0.0385 | 0.1923 |
| `r5g_b1_ratio70_30_s100_b0p03_lr5em6` | 0.8077 | 0.0769 | 0.0000 | 0.0769 | 0.1923 |
| `r5g_b2_ratio60_40_s100_b0p03_lr5em6` | 0.9615 | 0.0385 | 0.0000 | 0.0385 | 0.0385 |
| `r5g_b3_ratio50_50_s100_b0p03_lr5em6` | 0.9615 | 0.0385 | 0.0000 | 0.0385 | 0.0385 |
| `r5f2_main_from_r2c` | 0.9615 | 0.0000 | 0.0000 | 0.0385 | nan |
| `r5f2_real_only_from_r2c` | 0.5385 | 0.2308 | 0.0000 | 0.2308 | nan |

## R5G Success Rule

- low-to-high <= 0.4
- label2 recall >= 0.15
- D1 hidden pred>=4 <= 0.7
- high-to-low <= 0.04
- label5 recall >= 0.75
- MAE <= 0.5
- QWK >= 0.5

## Decision

- recommendation: `continue_risk_calibration`
- reason: No R5G run passes the risk-calibrated success rule.
- best_by_low_to_high_mae_qwk: `r5g_a3_real_only_s50_b0p05_lr5em6`
- passed_runs: none
- risk_frontier_runs: none

## Sources

- included prior rows: r2c_clean_reason_score_balanced, r4b_shuffled_reason_balanced, r5f2_main_from_r2c, r5f2_real_only_from_r2c
- missing prior rows: none
- missing new prediction runs: none

## Guardrails

- Evaluation uses the original dev split, not a balanced train distribution.
- Test split is not read.
- D1 annotations are evaluation references only.
- Human rationale is not included in the prediction prompt.
