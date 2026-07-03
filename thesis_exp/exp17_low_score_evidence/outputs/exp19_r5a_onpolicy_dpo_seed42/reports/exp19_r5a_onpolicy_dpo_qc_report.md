# Exp19-R5A Risk-Balanced Synthetic-Template DPO Scout QC Report

Exp19-R5A constructs DPO preference pairs from the train split only. It does not train, read test,
or use dev/D1 annotations for training.

R5A is **not strictly on-policy**: actual model-output rejected responses are limited, while
template rejected responses dominate. Treat R5A as a DPO pipeline scout/control, not as the main DPO
experiment.

## Summary

- total DPO pairs constructed: 3000
- actual model-output rejected responses: 171 (0.0570)
- template rejected responses: 2829 (0.9430)
- recovered human rationales checked for prompt leakage: 80
- exact human-rationale leakage count in prompts: 0
- low_to_high rejected validity rate: 1.0000
- high_to_low rejected validity rate: 1.0000
- dataset ready for DPO pipeline scout: `True`
- dataset ready for main DPO: `False`
- better name: `risk-balanced synthetic-template DPO scout`.
- R5B rejection-mined hybrid DPO should be used before claiming a main DPO result.

## Pair Counts

| risk_type | original | expanded | target | actual |
|---|---:|---:|---:|---:|
| low_to_high | 80 | 1200 | 0.4000 | 0.4000 |
| high_to_low_protection | 2659 | 1200 | 0.4000 | 0.4000 |
| mid_score_calibration | 297 | 600 | 0.2000 | 0.2000 |

## Rejected Source Counts

| source | risk_type | n |
|---|---|---:|
| r1b | high_to_low_protection | 1 |
| r2c | high_to_low_protection | 1 |
| r4b | high_to_low_protection | 4 |
| template_low_false_failure | high_to_low_protection | 1194 |
| r2n | low_to_high | 165 |
| template_high_no_failure | low_to_high | 1035 |
| template_extreme | mid_score_calibration | 600 |

## Pair Quality

| risk_type | n | chosen_mean | rejected_mean | gap | chosen_failure | rejected_failure | rejected_no_failure |
|---|---:|---:|---:|---:|---:|---:|---:|
| low_to_high | 1200 | 1.4392 | 4.9317 | 3.4925 | 1.0000 | 0.0000 | 1.0000 |
| high_to_low_protection | 1200 | 4.6550 | 1.9975 | 2.6575 | 0.0000 | 0.9975 | 0.0025 |
| mid_score_calibration | 600 | 3.0000 | 3.1733 | 2.0000 | 0.0000 | 0.4567 | 0.5433 |

## Guardrails

- DPO prompt contains only question, answer, metric, rubric, and metadata.
- Human-rationale-derived fields appear only in chosen assistant targets.
- Full DPO JSON is written under gitignored `data/` and should not be committed.
- Dev/D1 annotations are not read by this script and must remain evaluation-only.
