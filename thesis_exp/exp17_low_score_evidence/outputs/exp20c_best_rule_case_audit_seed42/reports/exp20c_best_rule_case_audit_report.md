# Exp20C Best-Rule Case Audit

Exp20C audits case-level effects of the Exp20B best automatic downgrade rule and
the best selective review rule. It uses existing dev predictions only, does not train,
and does not read test.

## Audited Rules

- automatic downgrade score model: `r5g_a3_real_only_s50_b0p05_lr5em6`
- automatic downgrade risk model: `r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6`
- automatic downgrade rule: `score_pred >= 5 and risk_pred <= 3 -> final_pred = 3`
- selective review score model: `r4b_shuffled_reason_balanced`
- selective review risk model: `r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6`
- selective review rule: `score_pred >= 4 and score_pred - risk_pred >= 1 -> review`

## Automatic Downgrade Effects

- flagged_count: 48
- flagged_rate: 0.0434
- rescued_low_to_high_count: 6
- rescued_low_to_high_rate: 0.2609
- gold_high_downgraded_to_3_count: 41
- gold_high_downgraded_to_3_rate: 0.0434
- d1_hidden_residual_pred_ge4_rate: 0.6154

## Score Metrics

| row | MAE | QWK | low-to-high | label2 recall | high-to-low | high-to-mid | label5 recall | D1 pred>=4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| score baseline | 0.4905 | 0.5010 | 0.4035 | 0.1579 | 0.0212 | 0.1058 | 0.7716 | 0.8077 |
| after downgrade | 0.5303 | 0.4836 | 0.2982 | 0.1579 | 0.0212 | 0.1492 | 0.7194 | 0.6154 |

## Selective Review Rule

- flagged_for_review_count: 343
- flagged_for_review_rate: 0.3098
- low-to-high recall among baseline errors: 0.7407
- gold-high flag rate: 0.3048
- D1 hidden residual if covered-only: 0.2692

## Decision

- recommendation: `d1_like_data_expansion_before_formal_rq3`
- reason: The rule is informative but leaves too much D1 hidden risk or insufficiently reliable interventions.
- recommend_automatic_downgrade_formal: False
- recommend_selective_review_formal: False
- recommend_data_expansion: True

## Guardrails

- Test split is not read.
- No model training is performed.
- D1 annotations are used only for evaluation.
- Human rationale is not used as decision input.
- Raw predictions are not written by this script.
