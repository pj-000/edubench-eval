# Exp23 R7 DPO Scout Dev Evaluation

Exp23 is an ordinary DPO sanity check for R7 data. The primary comparison is R7D
human-reason chosen vs R7E exactly matched score-only control.

## Dev Metrics

| run | dataset | MAE | QWK | low-to-high | high-to-low | label2 recall | label5 recall | D1 pred>=4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `r7d_reason_real_s100_b0p03_lr5em6` | `human_reason_real_error` | 0.5739 | 0.3258 | 0.6842 | 0.0455 | 0.2632 | 0.7986 | 0.6923 |
| `r7e_matched_score_only_s100_b0p03_lr5em6` | `matched_score_only_control` | 0.4959 | 0.3833 | 0.8596 | 0.0095 | 0.0789 | 0.5809 | 0.9615 |
| `r7f_score_reason_consistency_s100_b0p03_lr5em6` | `score_reason_consistency_counterfactual` | 0.4155 | 0.5040 | 0.5965 | 0.0000 | 0.0000 | 0.8399 | 1.0000 |
| `r2c_clean_reason_score_balanced` | `none_init_baseline` | 0.4219 | 0.4982 | 0.5965 | 0.0000 | 0.0000 | 0.8435 | 1.0000 |
| `r4b_shuffled_reason_balanced` | `none_init_baseline` | 0.4074 | 0.5617 | 0.4737 | 0.0063 | 0.0526 | 0.8363 | 0.8462 |

## Structured Failure Diagnostics

| run | failure micro-F1 | D1 nonempty failure | score_cap nonnull |
|---|---:|---:|---:|
| `r7d_reason_real_s100_b0p03_lr5em6` | 0.0000 | 0.0000 | 0.2692 |
| `r7e_matched_score_only_s100_b0p03_lr5em6` | 0.0000 | 0.0000 | 0.0385 |
| `r7f_score_reason_consistency_s100_b0p03_lr5em6` | 0.0000 | 0.0000 | 0.0000 |
| `r2c_clean_reason_score_balanced` | 0.0000 | 0.0000 | nan |
| `r4b_shuffled_reason_balanced` | 0.0000 | 0.0000 | nan |

## Primary Decision

- recommendation: `reason_dpo_not_yet_supported`
- reason: R7D does not clearly beat the exactly matched score-only R7E control under ordinary DPO.

R7F is an auxiliary consistency scout. It should not be used to claim that natural
reason-aware real-error DPO works unless R7D also beats R7E.

## Sources

- included baseline rows: r2c_clean_reason_score_balanced, r4b_shuffled_reason_balanced
- missing baseline rows: none
- missing Exp23 predictions: none

## Guardrails

- Test split is not read.
- Dev labels are used only for final dev evaluation, not for training.
- D1 annotations are evaluation references only.
- Raw predictions and logs should remain uncommitted.
