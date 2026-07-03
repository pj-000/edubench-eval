# Exp19-R5C Score-Risk DPO QC Report

This train-only DPO dataset isolates score-level risk. Human rationale is never included in the user
prompt.

## Summary

- r5c_main total pairs: 3000
- r5c_main actual model rejected: 189 (0.0630)
- r5c_main hard synthetic rejected: 2811 (0.9370)
- r5c_main extreme template rejected: 0 (0.0000)
- r5c_main score-risk validity: 1.0000
- r5c_main ready_for_main_score_risk_dpo: `True`
- r5c_no_mid total pairs: 3000
- r5c_no_mid actual model rejected: 243 (0.0810)
- r5c_no_mid hard synthetic rejected: 2757 (0.9190)
- r5c_no_mid extreme template rejected: 0 (0.0000)
- r5c_no_mid score-risk validity: 1.0000
- r5c_no_mid ready_for_main_score_risk_dpo: `True`
- recovered human rationales checked for prompt leakage: 80
- exact human-rationale leakage count in prompts: 0
- hard synthetic control mode: `False`
- generation mining needed: `True`

## Pair Counts

| variant | risk_type | original | expanded | actual_fraction |
|---|---|---:|---:|---:|
| r5c_main | high_to_low_score_risk | 2659 | 1350 | 0.4500 |
| r5c_main | low_to_high_score_risk | 80 | 1350 | 0.4500 |
| r5c_main | mid_score_calibration | 297 | 300 | 0.1000 |
| r5c_no_mid | high_to_low_score_risk | 2659 | 1500 | 0.5000 |
| r5c_no_mid | low_to_high_score_risk | 80 | 1500 | 0.5000 |

## Rejected Sources

| variant | source | category | risk_type | n |
|---|---|---|---|---:|
| r5c_main | r4b | actual_model | high_to_low_score_risk | 5 |
| r5c_main | hard_synthetic_score2_conservative_failure | hard_synthetic | high_to_low_score_risk | 468 |
| r5c_main | hard_synthetic_score3_conservative_failure | hard_synthetic | high_to_low_score_risk | 877 |
| r5c_main | r2n | actual_model | low_to_high_score_risk | 184 |
| r5c_main | hard_synthetic_score4_no_failure | hard_synthetic | low_to_high_score_risk | 1166 |
| r5c_main | hard_synthetic_mid_high5 | hard_synthetic | mid_score_calibration | 153 |
| r5c_main | hard_synthetic_mid_low1 | hard_synthetic | mid_score_calibration | 147 |
| r5c_no_mid | r1b | actual_model | high_to_low_score_risk | 2 |
| r5c_no_mid | r2c | actual_model | high_to_low_score_risk | 1 |
| r5c_no_mid | r4b | actual_model | high_to_low_score_risk | 2 |
| r5c_no_mid | hard_synthetic_score2_conservative_failure | hard_synthetic | high_to_low_score_risk | 523 |
| r5c_no_mid | hard_synthetic_score3_conservative_failure | hard_synthetic | high_to_low_score_risk | 972 |
| r5c_no_mid | r2n | actual_model | low_to_high_score_risk | 238 |
| r5c_no_mid | hard_synthetic_score4_no_failure | hard_synthetic | low_to_high_score_risk | 1262 |

## Pair Quality

| variant | risk_type | n | chosen_mean | rejected_mean | gap | actual_rate | hard_rate | validity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r5c_main | high_to_low_score_risk | 1350 | 4.6637 | 2.6467 | 2.0170 | 0.0037 | 0.9963 | 1.0000 |
| r5c_main | low_to_high_score_risk | 1350 | 1.4274 | 4.0659 | 2.6385 | 0.1363 | 0.8637 | 1.0000 |
| r5c_main | mid_score_calibration | 300 | 3.0000 | 3.0400 | 2.0000 | 0.0000 | 1.0000 | 1.0000 |
| r5c_no_mid | high_to_low_score_risk | 1500 | 4.6513 | 2.6460 | 2.0053 | 0.0033 | 0.9967 | 1.0000 |
| r5c_no_mid | low_to_high_score_risk | 1500 | 1.4500 | 4.0753 | 2.6253 | 0.1587 | 0.8413 | 1.0000 |

## Decision

- R5C_main uses 45/45/10 low/high/mid score-risk ratios.
- R5C_no_mid uses 50/50 low/high score-risk ratios.
- R5E hard-synthetic control uses the same ratio logic without actual model rejected responses.
- Full DPO JSON is gitignored; only QC artifacts should be committed.
