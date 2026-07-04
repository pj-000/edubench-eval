# Exp20 Dual-Model Risk-Aware Gate Evaluation

Exp20 evaluates decision-layer gates using existing dev predictions only. It does not train
and does not read test.

## Models

- score models: r4b_shuffled_reason_balanced, r5g_a3_real_only_s50_b0p05_lr5em6, r5h_h6_from_r5g_a3_highprotect_s30_b0p02_lr1e6
- risk models: r5f2_real_only_from_r2c, r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6, r5h_h2_from_r5f2_real_highprotect_s20_b0p02_lr1e6

## Best Abstention Rule

- score_model: `r5g_a3_real_only_s50_b0p05_lr5em6`
- risk_model: `r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6`
- rule: `gap_abstain`
- coverage: 0.9313
- low_to_high_covered: 0.3200
- D1_hidden_pred_ge4_covered: 0.7500
- flag_rate_on_gold_high: 0.0667

## Best Downgrade Rule

- score_model: `r4b_shuffled_reason_balanced`
- risk_model: `r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6`
- rule: `gap_downgrade_to_3`
- MAE: 0.4842
- QWK: 0.5247
- low_to_high: 0.2807
- label2_recall: 0.0526
- high_to_low: 0.0063
- label5_recall: 0.7518
- D1_hidden_pred_ge4: 0.4615

## Does The Risk Model Flag Dangerous High-Score Predictions?

| score | risk | rule | flagged | precision | recall |
|---|---|---|---:|---:|---:|
| `r4b_shuffled_reason_balanced` | `r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6` | `gap_abstain` | 110 | 0.1000 | 0.4074 |
| `r4b_shuffled_reason_balanced` | `r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6` | `gap_downgrade_to_3` | 110 | 0.1000 | 0.4074 |
| `r4b_shuffled_reason_balanced` | `r5f2_real_only_from_r2c` | `gap_abstain` | 117 | 0.0940 | 0.4074 |
| `r4b_shuffled_reason_balanced` | `r5f2_real_only_from_r2c` | `gap_downgrade_to_3` | 117 | 0.0940 | 0.4074 |
| `r4b_shuffled_reason_balanced` | `r5h_h2_from_r5f2_real_highprotect_s20_b0p02_lr1e6` | `gap_abstain` | 102 | 0.0980 | 0.3704 |

## Decision

- recommendation: `d1_like_data_expansion_not_more_dpo`
- reason: Both abstention and downgrade fail the current success criteria.

## Required Questions

- Does the risk model flag dangerous high-score predictions? See `exp20_gate_flag_analysis.csv`.
- Is abstention better than automatic downgrade? Compare the decision and the best-rule tables above.
- What coverage is needed to reduce low-to-high below 0.30? See `exp20_gate_abstention_frontier.csv`.
- Does gate over-flag true high scores? Check `flag_rate_on_gold_high` and `flagged_gold_high_rate`.
- Which score/risk pair is best? See `best_abstention_rule` and `best_downgrade_rule` in decision JSON.
- Should RQ3 use selective review rather than more DPO? The decision JSON states the recommendation.

## Guardrails

- Test split is not read.
- No model training is performed.
- D1 annotations are used only for evaluation.
- Human rationale is not used as decision input.
- Raw predictions remain local/server-side and are not written by this script.
