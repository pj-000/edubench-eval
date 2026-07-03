# Exp19-R5 DPO Scout Dev Evaluation

This report summarizes small-step DPO scout adapters on the original question-disjoint dev split.
Raw predictions/logs/checkpoints remain gitignored. No test split is read.

## Dev Metrics

| run | init | dataset | MAE | QWK | low-to-high | high-to-low | label2 recall |
|---|---|---|---:|---:|---:|---:|---:|
| `r5c_from_r2c` | `r2c_clean_reason_score_balanced` | `r5c_score_risk` | 0.4544 | 0.4860 | 0.5263 | 0.0138 | 0.0263 |
| `r5c_from_r1b` | `r1b_score_only_balanced` | `r5c_score_risk` | 0.3866 | 0.5714 | 0.4912 | 0.0021 | 0.0263 |
| `r5d_from_r2c` | `r2c_clean_reason_score_balanced` | `r5d_evidence_consistency` | 0.4128 | 0.4913 | 0.7018 | 0.0000 | 0.0263 |
| `r5e_from_r2c` | `r2c_clean_reason_score_balanced` | `r5e_hard_synthetic_control` | 0.4535 | 0.4921 | 0.5439 | 0.0085 | 0.0263 |
| `r1b_score_only_balanced` | `r1b_score_only_balanced` | `none_init_baseline` | 0.3975 | 0.5565 | 0.5263 | 0.0021 | 0.0000 |
| `r2c_clean_reason_score_balanced` | `r2c_clean_reason_score_balanced` | `none_init_baseline` | 0.4219 | 0.4982 | 0.5965 | 0.0000 | 0.0000 |
| `r4b_shuffled_reason_balanced` | `r4b_shuffled_reason_balanced` | `none_init_baseline` | 0.4074 | 0.5617 | 0.4737 | 0.0063 | 0.0526 |

## Required Questions

- Does R5C from R2c reduce low-to-high vs R2c: yes, delta=0.0702
- Does R5C from R1b reduce low-to-high vs R1b: yes, delta=0.0351
- Does R5D improve structured failure outputs: no, nonempty_delta=0.0000, micro_f1_delta=0.0000
- Does R5E hard-synthetic perform similarly to R5C: yes; if R5C is not clearly better than R5E, template effects remain a concern (low_to_high_gap=0.0175)
- Does any DPO run damage high-score protection: not by the >0.05 rule
- Should we run full DPO: need_more_rejection_mining
- Which dataset/init pair is most promising: none locked yet

## D1 Hidden And Failure-Type Tables

| run | D1 pred>=4 | D1 label2 recall | failure micro-F1 | D1 nonempty failure | score_cap nonnull |
|---|---:|---:|---:|---:|---:|
| `r5c_from_r2c` | 0.9615 | 0.0385 | 0.0000 | 0.0385 | 0.0385 |
| `r5c_from_r1b` | 0.9231 | 0.0385 | 0.0000 | 0.0000 | 0.0000 |
| `r5d_from_r2c` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `r5e_from_r2c` | 0.9615 | 0.0385 | 0.0000 | 0.0385 | 0.0385 |
| `r1b_score_only_balanced` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | nan |
| `r2c_clean_reason_score_balanced` | 1.0000 | 0.0000 | 0.0000 | 0.0000 | nan |
| `r4b_shuffled_reason_balanced` | 0.8462 | 0.0385 | 0.0000 | 0.0000 | nan |

## Baselines

- included baselines: r1b_score_only_balanced, r2c_clean_reason_score_balanced, r4b_shuffled_reason_balanced
- missing baselines: none
- missing DPO prediction runs: none

## Decision

- decision: `need_more_rejection_mining`
- reason: There is some risk-side movement, but the scout does not meet the full success rule.

## Guardrails

- Evaluation uses the original dev split, not balanced train distribution.
- Test split is not read.
- D1 annotations are evaluation references only.
- Human rationale is not included in the prediction prompt.
