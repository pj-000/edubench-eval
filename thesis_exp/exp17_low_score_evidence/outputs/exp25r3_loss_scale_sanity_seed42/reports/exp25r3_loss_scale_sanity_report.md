# Exp25R3 Loss-Scale + Field-Masked SRC-DPO Sanity

This is a train-only sanity check. It does not read dev/test and does not generate predictions.

## Overall

| config | mode | beta | pref_ftx | n | before acc | after acc | acc gain | first-step delta gain | mean delta gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mixed_field_b1_ftx0` | field | 1.0 | 0.0 | 64 | 0.0000 | 0.8906 | 0.8906 | 0.4531 | 0.3998 |
| `mixed_field_b3_ftx0` | field | 3.0 | 0.0 | 64 | 0.0000 | 0.8438 | 0.8438 | 0.6094 | 0.3942 |
| `score_field_b1_ftx0` | field | 1.0 | 0.0 | 32 | 0.0000 | 0.9688 | 0.9688 | 0.1875 | 0.6334 |
| `score_field_b3_ftx0` | field | 3.0 | 0.0 | 32 | 0.0000 | 1.0000 | 1.0000 | 0.2266 | 0.6351 |
| `score_full_b0p03_ftx0p05` | full | 0.03 | 0.05 | 32 | 0.0000 | 0.1875 | 0.1875 | -0.0312 | -0.0122 |
| `score_full_b1_ftx0` | full | 1.0 | 0.0 | 32 | 0.0000 | 0.7812 | 0.7812 | 0.0938 | 0.0522 |

## After Training By Negative Type

| config | negative_type | n | dpo pref acc | mean delta |
|---|---|---:|---:|---:|
| `mixed_field_b1_ftx0` | `high_protection_score_mismatch` | 16 | 1.0000 | 0.5422 |
| `mixed_field_b1_ftx0` | `low_failure_erasure_counterfactual` | 16 | 1.0000 | 0.4609 |
| `mixed_field_b1_ftx0` | `reason_mismatch_same_score` | 16 | 0.5625 | 0.0537 |
| `mixed_field_b1_ftx0` | `score_mismatch_same_reason` | 16 | 1.0000 | 0.5422 |
| `mixed_field_b3_ftx0` | `high_protection_score_mismatch` | 16 | 1.0000 | 0.5505 |
| `mixed_field_b3_ftx0` | `low_failure_erasure_counterfactual` | 16 | 1.0000 | 0.4492 |
| `mixed_field_b3_ftx0` | `reason_mismatch_same_score` | 16 | 0.3750 | 0.0264 |
| `mixed_field_b3_ftx0` | `score_mismatch_same_reason` | 16 | 1.0000 | 0.5505 |
| `score_field_b1_ftx0` | `score_mismatch_same_reason` | 32 | 0.9688 | 0.6334 |
| `score_field_b3_ftx0` | `score_mismatch_same_reason` | 32 | 1.0000 | 0.6351 |
| `score_full_b0p03_ftx0p05` | `score_mismatch_same_reason` | 32 | 0.1875 | -0.0122 |
| `score_full_b1_ftx0` | `score_mismatch_same_reason` | 32 | 0.7812 | 0.0522 |

## Decision

- recommendation: `field_mask_scale_sanity_passed_run_corrected_dev_scout`
- reason: At least one field-masked config overfit train preferences and score mismatch improved.

## Guardrails

- No dev/test split is read.
- No generated prediction JSONL is written.
- This experiment is not a formal dev result.
