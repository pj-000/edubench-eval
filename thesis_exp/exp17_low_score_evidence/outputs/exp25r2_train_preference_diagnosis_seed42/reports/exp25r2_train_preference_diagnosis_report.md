# Exp25R2 Train Preference Diagnosis

This diagnostic checks whether Exp25 trained adapters prefer chosen over rejected on train-only DPO
pairs.
It does not train, generate text, or read test labels.

## Overall

| run | n | dpo pref acc | raw pref acc | mean delta | p50 delta | unique source samples |
|---|---:|---:|---:|---:|---:|---:|
| `exp25_src_score_mismatch_r2c` | 429 | 0.0816 | 0.9604 | -0.0434 | -0.0312 | 314 |
| `exp25_src_mixed_r2c` | 1127 | 0.2334 | 0.8776 | -0.0534 | -0.0469 | 314 |

## By Negative Type

| run | negative_type | n | dpo pref acc | mean delta |
|---|---|---:|---:|---:|
| `exp25_src_mixed_r2c` | `high_protection_score_mismatch` | 139 | 0.0576 | -0.0482 |
| `exp25_src_mixed_r2c` | `low_failure_erasure_counterfactual` | 130 | 0.2000 | -0.2778 |
| `exp25_src_mixed_r2c` | `reason_mismatch_same_score` | 429 | 0.4732 | 0.0080 |
| `exp25_src_mixed_r2c` | `score_mismatch_same_reason` | 429 | 0.0606 | -0.0484 |
| `exp25_src_score_mismatch_r2c` | `score_mismatch_same_reason` | 429 | 0.0816 | -0.0434 |

## By Risk Type

| run | risk_type | n | dpo pref acc | mean delta |
|---|---|---:|---:|---:|
| `exp25_src_mixed_r2c` | `high_to_low_real_model_error` | 417 | 0.1942 | -0.0357 |
| `exp25_src_mixed_r2c` | `high_to_mid_real_model_error` | 262 | 0.2672 | -0.0428 |
| `exp25_src_mixed_r2c` | `low_to_high_real_model_error` | 390 | 0.2462 | -0.0918 |
| `exp25_src_mixed_r2c` | `low_to_mid_real_model_error` | 58 | 0.2759 | 0.0306 |
| `exp25_src_score_mismatch_r2c` | `high_to_low_real_model_error` | 139 | 0.0647 | -0.0474 |
| `exp25_src_score_mismatch_r2c` | `high_to_mid_real_model_error` | 131 | 0.1374 | -0.0220 |
| `exp25_src_score_mismatch_r2c` | `low_to_high_real_model_error` | 130 | 0.0538 | -0.0602 |
| `exp25_src_score_mismatch_r2c` | `low_to_mid_real_model_error` | 29 | 0.0345 | -0.0458 |

## Decision

- recommendation: `fix_loss_beta_steps_or_trainer_before_data_expansion`
- reason: At least one Exp25 run has train DPO preference accuracy below 0.60.

## Guardrails

- No test split is read.
- No training is performed.
- Human reason remains only in assistant targets from train-only DPO pairs.
- Raw generated predictions are not written by this diagnostic.
