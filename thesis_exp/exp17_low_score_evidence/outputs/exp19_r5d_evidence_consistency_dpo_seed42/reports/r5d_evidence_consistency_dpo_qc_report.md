# Exp19-R5D Evidence-Consistency DPO QC Report

R5D trains structured score/failure consistency rather than score-level risk. It is an auxiliary DPO
dataset.

## Summary

- total DPO pairs constructed: 3000
- actual model-output rejected responses: 1530 (0.5100)
- hard synthetic rejected responses: 1470 (0.4900)
- recovered human rationales checked for prompt leakage: 80
- exact human-rationale leakage count in prompts: 0
- consistency validity rate: 1.0000
- ready_for_evidence_consistency_dpo: `True`

## Pair Counts

| risk_type | original | expanded | actual_fraction |
|---|---:|---:|---:|
| low_failure_evidence_consistency | 108 | 1500 | 0.5000 |
| high_no_false_failure_consistency | 2719 | 1500 | 0.5000 |

## Rejected Sources

| source | category | risk_type | n |
|---|---|---|---:|
| r1b | actual_model | high_no_false_failure_consistency | 31 |
| r2c | actual_model | high_no_false_failure_consistency | 17 |
| r2n | actual_model | high_no_false_failure_consistency | 15 |
| r4b | actual_model | high_no_false_failure_consistency | 23 |
| hard_synthetic_high_false_failure_score3 | hard_synthetic | high_no_false_failure_consistency | 572 |
| hard_synthetic_high_false_failure_score4 | hard_synthetic | high_no_false_failure_consistency | 842 |
| r2n | actual_model | low_failure_evidence_consistency | 449 |
| r4b | actual_model | low_failure_evidence_consistency | 995 |
| hard_synthetic_low_no_failure_inconsistent | hard_synthetic | low_failure_evidence_consistency | 56 |

## Pair Quality

| risk_type | n | chosen_mean | rejected_mean | actual_rate | hard_rate | validity | rejected_failure | rejected_no_failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| low_failure_evidence_consistency | 1500 | 1.5273 | 1.8567 | 0.9627 | 0.0373 | 1.0000 | 0.0000 | 0.6920 |
| high_no_false_failure_consistency | 1500 | 4.6467 | 3.5613 | 0.0573 | 0.9427 | 1.0000 | 0.5613 | 0.0153 |

## Evaluation Focus

- major_failure_nonempty_rate on D1 hidden cases
- no_major_failure rate on clean high controls
- failure type F1
- score_cap and rubric_satisfied consistency

Full DPO JSON is gitignored and must not be committed.
