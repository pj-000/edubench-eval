# Exp19-R5B Rejection-Mined Hybrid DPO QC Report

Exp19-R5B constructs train-only DPO preference pairs by prioritizing actual SFT model mistakes as
rejected responses. Hard synthetic fallback is used when real mistakes are unavailable; extreme
templates are last fallback.

## Summary

- total DPO pairs constructed: 3000
- actual model-output rejected responses: 1203 (0.4010)
- hard synthetic rejected responses: 1797 (0.5990)
- extreme template rejected responses: 0 (0.0000)
- recovered human rationales checked for prompt leakage: 80
- exact human-rationale leakage count in prompts: 0
- low_to_high rejected validity rate: 1.0000
- high_to_low rejected validity rate: 1.0000
- need more rejection generation: `False`
- ready_for_main_dpo: `True`

## Pair Counts

| risk_type | original | expanded | target | actual |
|---|---:|---:|---:|---:|
| low_to_high | 90 | 1200 | 0.4000 | 0.4000 |
| high_to_low_protection | 2723 | 1200 | 0.4000 | 0.4000 |
| mid_score_calibration | 297 | 600 | 0.2000 | 0.2000 |

## Rejected Source Counts

| source | category | risk_type | n |
|---|---|---|---:|
| r1b | actual_model | high_to_low_protection | 35 |
| r2c | actual_model | high_to_low_protection | 18 |
| r2n | actual_model | high_to_low_protection | 15 |
| r4b | actual_model | high_to_low_protection | 28 |
| hard_synthetic_conservative_insufficient | hard_synthetic | high_to_low_protection | 778 |
| hard_synthetic_conservative_unclear | hard_synthetic | high_to_low_protection | 326 |
| r2n | actual_model | low_to_high | 153 |
| r4b | actual_model | low_to_high | 954 |
| hard_synthetic_high_no_failure | hard_synthetic | low_to_high | 60 |
| hard_synthetic_high_unclear | hard_synthetic | low_to_high | 33 |
| hard_synthetic_mid_high | hard_synthetic | mid_score_calibration | 290 |
| hard_synthetic_mid_low | hard_synthetic | mid_score_calibration | 310 |

## Pair Quality

| risk_type | n | chosen_mean | rejected_mean | gap | chosen_failure | rejected_failure | rejected_no_failure |
|---|---:|---:|---:|---:|---:|---:|---:|
| low_to_high | 1200 | 1.4608 | 2.0292 | 0.5683 | 1.0000 | 0.0000 | 0.9725 |
| high_to_low_protection | 1200 | 4.6342 | 2.7225 | 1.9117 | 0.0000 | 0.6517 | 0.0217 |
| mid_score_calibration | 600 | 3.0000 | 2.9333 | 2.0000 | 0.0000 | 0.5167 | 0.4833 |

## Hardness

| risk_type | category | n | within_risk | overall | gap | validity |
|---|---|---:|---:|---:|---:|---:|
| low_to_high | actual_model | 1107 | 0.9225 | 0.3690 | 0.3884 | 1.0000 |
| low_to_high | hard_synthetic | 93 | 0.0775 | 0.0310 | 2.7097 | 1.0000 |
| high_to_low_protection | actual_model | 96 | 0.0800 | 0.0320 | 1.3438 | 1.0000 |
| high_to_low_protection | hard_synthetic | 1104 | 0.9200 | 0.3680 | 1.9611 | 1.0000 |
| mid_score_calibration | hard_synthetic | 600 | 1.0000 | 0.2000 | 2.0000 | 1.0000 |

## Decision

- This dataset is hybrid, not purely on-policy.
- If actual model rejected rate remains below 0.20, use it only as a DPO scout/control.
- If more generations raise actual rejected rate above 0.30, it can be reconsidered as the main DPO dataset.

## Guardrails

- The DPO prompt contains only question, answer, metric, rubric, and metadata.
- Human-rationale-derived fields appear only in chosen assistant targets.
- Full DPO JSON is written under gitignored `data/` and must not be committed.
- Test is not read. Dev/D1 annotations are not used as training labels.
