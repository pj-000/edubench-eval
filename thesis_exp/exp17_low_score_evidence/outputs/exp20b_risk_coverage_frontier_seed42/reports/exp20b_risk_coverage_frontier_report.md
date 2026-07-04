# Exp20B Risk-Coverage Frontier

Exp20B searches high-score gate thresholds over existing dev predictions only. It does not train
and does not read test.

## Grid

- score models: r4b_shuffled_reason_balanced, r5g_a3_real_only_s50_b0p05_lr5em6, r5h_h6_from_r5g_a3_highprotect_s30_b0p02_lr1e6
- risk models: r5f2_real_only_from_r2c, r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6, r5h_h2_from_r5f2_real_highprotect_s20_b0p02_lr1e6
- rule specs per score/risk pair: 58
- abstention rows: 522
- downgrade rows: 522
- pareto rows: 103

## Best Selective Review Rule

- score_model: `r4b_shuffled_reason_balanced`
- risk_model: `r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6`
- rule: `gap_ge`
- thresholds: `{"gap_threshold": 1, "risk_threshold": null, "score_cap_threshold": null, "score_threshold": 4}`
- coverage: 0.6902
- low_to_high_covered: 0.1892
- D1_hidden_pred_ge4_covered: 0.6364
- flag_rate_on_gold_high: 0.3048

## Best Automatic Downgrade Rule

- score_model: `r5g_a3_real_only_s50_b0p05_lr5em6`
- risk_model: `r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6`
- rule: `risk_pred_le`
- thresholds: `{"gap_threshold": null, "risk_threshold": 3, "score_cap_threshold": null, "score_threshold": 5}`
- MAE: 0.5303
- QWK: 0.4836
- low_to_high: 0.2982
- high_to_low: 0.0212
- label5_recall: 0.7194
- D1_hidden_pred_ge4: 0.6154

## Required Questions

- Selective rule with coverage>=0.85 and low_to_high_covered<=0.30: False.
  No. Rules that reduce low-to-high enough either miss the D1 hidden target or sacrifice too much
coverage.
- Rule reducing D1_hidden_pred_ge4_covered<=0.50: False.
- Downgrade vs abstention: Downgrade is stronger under the current grid criteria, but it is an automatic score intervention.
- Best coverage-risk tradeoff: see the best selective review rule above and the Pareto table.
- High-score over-flagging: For selective review, the best low-risk rule over-flags true high scores.
- Downgrade rules passing the low-risk/high-protection screen before MAE/QWK deltas: 204.
- RQ3 selective review recommendation: False.
- RQ3 automatic downgrade recommendation: True.
- Data expansion recommendation: False.

## Decision

- reason: Automatic downgrade has a successful frontier rule while selective review does not.

## Guardrails

- Test split is not read.
- No model training is performed.
- D1 annotations are used only for evaluation.
- Human rationale is not used as decision input.
- Raw predictions remain local/server-side and are not written by this script.
