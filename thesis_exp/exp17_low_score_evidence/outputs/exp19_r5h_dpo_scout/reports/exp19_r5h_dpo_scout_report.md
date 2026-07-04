# Exp19-R5H Two-Stage DPO Scout Dev Evaluation

R5H starts from low-risk DPO adapters and applies lightweight high-protection-only DPO.

## Dev Metrics

| run | init | dataset | MAE | QWK | low-to-high | high-to-low | label2 recall | label5 recall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6` | `r5f2_real_only_from_r2c` | `r5h_high_protection_only` | 0.6495 | 0.4207 | 0.2632 | 0.0794 | 0.3158 | 0.6745 |
| `r5h_h2_from_r5f2_real_highprotect_s20_b0p02_lr1e6` | `r5f2_real_only_from_r2c` | `r5h_high_protection_only` | 0.6378 | 0.4207 | 0.2807 | 0.0762 | 0.3158 | 0.6799 |
| `r5h_h3_from_r5f2_real_highprotect_s30_b0p02_lr1e6` | `r5f2_real_only_from_r2c` | `r5h_high_protection_only` | 0.6188 | 0.4338 | 0.2982 | 0.0709 | 0.3158 | 0.6888 |
| `r5h_h4_from_r5f2_real_highprotect_s20_b0p03_lr2e6` | `r5f2_real_only_from_r2c` | `r5h_high_protection_only` | 0.6052 | 0.4384 | 0.2982 | 0.0677 | 0.3158 | 0.6960 |
| `r5h_h5_from_r5g_a3_highprotect_s20_b0p02_lr1e6` | `r5g_a3` | `r5h_high_protection_only` | 0.4761 | 0.5061 | 0.4386 | 0.0212 | 0.1316 | 0.7860 |
| `r5h_h6_from_r5g_a3_highprotect_s30_b0p02_lr1e6` | `r5g_a3` | `r5h_high_protection_only` | 0.4670 | 0.5104 | 0.4386 | 0.0159 | 0.1316 | 0.7860 |
| `r2c_clean_reason_score_balanced` | `r2c_clean_reason_score_balanced` | `none_init_baseline` | 0.4219 | 0.4982 | 0.5965 | 0.0000 | 0.0000 | 0.8435 |
| `r4b_shuffled_reason_balanced` | `r4b_shuffled_reason_balanced` | `none_init_baseline` | 0.4074 | 0.5617 | 0.4737 | 0.0063 | 0.0526 | 0.8363 |
| `r5f2_real_only_from_r2c` | `r2c_clean_reason_score_balanced` | `r5f2_real_only_small` | 0.6748 | 0.4031 | 0.2632 | 0.0847 | 0.3158 | 0.6619 |
| `r5g_a3_real_only_s50_b0p05_lr5em6` | `r2c_clean_reason_score_balanced` | `r5g_real_only` | 0.4905 | 0.5010 | 0.4035 | 0.0212 | 0.1579 | 0.7716 |

## D1 Hidden And Failure-Type Tables

| run | D1 pred>=4 | D1 label2 recall | failure micro-F1 | D1 nonempty failure | score_cap nonnull |
|---|---:|---:|---:|---:|---:|
| `r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6` | 0.5385 | 0.2308 | 0.0000 | 0.2308 | 0.4615 |
| `r5h_h2_from_r5f2_real_highprotect_s20_b0p02_lr1e6` | 0.5769 | 0.2308 | 0.0000 | 0.2308 | 0.4231 |
| `r5h_h3_from_r5f2_real_highprotect_s30_b0p02_lr1e6` | 0.6154 | 0.2308 | 0.0000 | 0.2308 | 0.3846 |
| `r5h_h4_from_r5f2_real_highprotect_s20_b0p03_lr2e6` | 0.6154 | 0.2308 | 0.0000 | 0.2308 | 0.3846 |
| `r5h_h5_from_r5g_a3_highprotect_s20_b0p02_lr1e6` | 0.8462 | 0.0385 | 0.0000 | 0.0385 | 0.1538 |
| `r5h_h6_from_r5g_a3_highprotect_s30_b0p02_lr1e6` | 0.8462 | 0.0385 | 0.0000 | 0.0385 | 0.1538 |
| `r5f2_real_only_from_r2c` | 0.5385 | 0.2308 | 0.0000 | 0.2308 | nan |
| `r5g_a3_real_only_s50_b0p05_lr5em6` | 0.8077 | 0.0769 | 0.0000 | 0.0385 | nan |

## R5H Success Rule

- low-to-high <= 0.35
- label2 recall >= 0.2
- D1 hidden pred>=4 <= 0.65
- high-to-low <= 0.04
- label5 recall >= 0.75
- MAE <= 0.5
- QWK >= 0.5

## Decision

- recommendation: `continue_or_stop_by_tradeoff`
- reason: low risk remains controlled but high-protection is too weak.
- best_by_low_high_mae_qwk: `r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6`
- passed_runs: none

## Sources

- included prior rows: r2c_clean_reason_score_balanced, r4b_shuffled_reason_balanced, r5f2_real_only_from_r2c, r5g_a3_real_only_s50_b0p05_lr5em6
- missing prior rows: none
- missing R5H prediction runs: none

## Guardrails

- Evaluation uses the original dev split.
- Test split is not read.
- D1 annotations are evaluation references only.
- Human rationale is not included in the prediction prompt.
