# Exp23 R7 DPO Scout Dev Evaluation

Exp23 is an ordinary DPO sanity check for R7 data. The primary comparison is R7D
human-reason chosen vs R7E exactly matched score-only control.

## Dev Metrics

| run | dataset | MAE | QWK | low-to-high | high-to-low | label2 recall | label5 recall | D1 pred>=4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `r2c_clean_reason_score_balanced` | `none_init_baseline` | 0.4219 | 0.4982 | 0.5965 | 0.0000 | 0.0000 | 0.8435 | 1.0000 |
| `r4b_shuffled_reason_balanced` | `none_init_baseline` | 0.4074 | 0.5617 | 0.4737 | 0.0063 | 0.0526 | 0.8363 | 0.8462 |

## Structured Failure Diagnostics

| run | failure micro-F1 | D1 nonempty failure | score_cap nonnull |
|---|---:|---:|---:|
| `r2c_clean_reason_score_balanced` | 0.0000 | 0.0000 | nan |
| `r4b_shuffled_reason_balanced` | 0.0000 | 0.0000 | nan |

## Primary Decision

- recommendation: `wait_for_primary_predictions`
- reason: R7D and R7E predictions are both required for the primary Exp23 comparison.

R7F is an auxiliary consistency scout. It should not be used to claim that natural
reason-aware real-error DPO works unless R7D also beats R7E.

## Sources

- included baseline rows: r2c_clean_reason_score_balanced, r4b_shuffled_reason_balanced
- missing baseline rows: none
- missing Exp23 predictions: r7d_reason_real_s100_b0p03_lr5em6, r7e_matched_score_only_s100_b0p03_lr5em6, r7f_score_reason_consistency_s100_b0p03_lr5em6

## Guardrails

- Test split is not read.
- Dev labels are used only for final dev evaluation, not for training.
- D1 annotations are evaluation references only.
- Raw predictions and logs should remain uncommitted.
