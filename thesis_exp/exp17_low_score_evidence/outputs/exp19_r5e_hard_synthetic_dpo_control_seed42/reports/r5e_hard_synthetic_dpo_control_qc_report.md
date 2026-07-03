# Exp19-R5E Hard-Synthetic DPO Control QC Report

This train-only DPO dataset isolates score-level risk. Human rationale is never included in the user
prompt.

## Summary

- r5e_main total pairs: 3000
- r5e_main actual model rejected: 0 (0.0000)
- r5e_main hard synthetic rejected: 3000 (1.0000)
- r5e_main extreme template rejected: 0 (0.0000)
- r5e_main score-risk validity: 1.0000
- r5e_main ready_for_main_score_risk_dpo: `True`
- recovered human rationales checked for prompt leakage: 80
- exact human-rationale leakage count in prompts: 0
- hard synthetic control mode: `True`
- generation mining needed: `False`

## Pair Counts

| variant | risk_type | original | expanded | actual_fraction |
|---|---|---:|---:|---:|
| r5e_main | high_to_low_score_risk | 2656 | 1350 | 0.4500 |
| r5e_main | low_to_high_score_risk | 80 | 1350 | 0.4500 |
| r5e_main | mid_score_calibration | 297 | 300 | 0.1000 |

## Rejected Sources

| variant | source | category | risk_type | n |
|---|---|---|---|---:|
| r5e_main | hard_synthetic_score2_conservative_failure | hard_synthetic | high_to_low_score_risk | 477 |
| r5e_main | hard_synthetic_score3_conservative_failure | hard_synthetic | high_to_low_score_risk | 873 |
| r5e_main | hard_synthetic_score4_no_failure | hard_synthetic | low_to_high_score_risk | 1350 |
| r5e_main | hard_synthetic_mid_high5 | hard_synthetic | mid_score_calibration | 140 |
| r5e_main | hard_synthetic_mid_low1 | hard_synthetic | mid_score_calibration | 160 |

## Pair Quality

| variant | risk_type | n | chosen_mean | rejected_mean | gap | actual_rate | hard_rate | validity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r5e_main | high_to_low_score_risk | 1350 | 4.6756 | 2.6467 | 2.0289 | 0.0000 | 1.0000 | 1.0000 |
| r5e_main | low_to_high_score_risk | 1350 | 1.4274 | 4.0000 | 2.5726 | 0.0000 | 1.0000 | 1.0000 |
| r5e_main | mid_score_calibration | 300 | 3.0000 | 2.8667 | 2.0000 | 0.0000 | 1.0000 | 1.0000 |

## Decision

- R5C_main uses 45/45/10 low/high/mid score-risk ratios.
- R5C_no_mid uses 50/50 low/high score-risk ratios.
- R5E hard-synthetic control uses the same ratio logic without actual model rejected responses.
- Full DPO JSON is gitignored; only QC artifacts should be committed.
