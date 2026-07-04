# Exp19-R5H Two-Stage DPO Data QC

R5H builds a stage-2 high-protection-only DPO dataset. It is intended to start from a
low-risk adapter and lightly restore high-score protection.

## Dataset

- dataset: `edubench_r5h_high_protection_only_dpo_train`
- pair_count: 555
- unique_source_sample_ids: 555
- chosen_score_mean: 4.6523
- rejected_score_mean: 2.3279
- rejected_false_failure_rate: 1.0000
- contains_low_to_high_pairs: `False`
- contains_dev_or_test_marker: `False`

## Source Files

| source | exists | rows | matching high-protection pairs | new unique pairs |
|---|---:|---:|---:|---:|
| `thesis_exp/exp17_low_score_evidence/outputs/exp19_r5f2_rejection_mining_seed42/data/edubench_r5f2_score_risk_main_dpo_train.json` | True | 1110 | 555 | 555 |
| `thesis_exp/exp17_low_score_evidence/outputs/exp19_r5g_risk_calibrated_dpo_seed42/data/edubench_r5g_ratio_70_30_dpo_train.json` | True | 793 | 238 | 0 |
| `thesis_exp/exp17_low_score_evidence/outputs/exp19_r5g_risk_calibrated_dpo_seed42/data/edubench_r5g_ratio_60_40_dpo_train.json` | True | 925 | 370 | 0 |
| `thesis_exp/exp17_low_score_evidence/outputs/exp19_r5g_risk_calibrated_dpo_seed42/data/edubench_r5g_ratio_50_50_dpo_train.json` | True | 1110 | 555 | 0 |

## Guardrails

- This preparation step does not read test.
- No human rationale text is added to the prompt.
- Full DPO JSON is written under gitignored `data/`.
- R5H stage-2 data contains only high-protection pairs.
