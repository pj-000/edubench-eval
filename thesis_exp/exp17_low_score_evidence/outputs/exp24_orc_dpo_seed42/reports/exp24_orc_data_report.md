# Exp24 ORC-DPO Score-Channel Data Preparation

R7G is derived from the exact R7D/R7E matched 429-pair pool.

## Key Design

- DPO main channel is score-only: `{"score": gold}` vs `{"score": rejected}`.
- Human rationale is stored as `auxiliary_reason_target` and is not included in the prompt.
- `risk_type` and `ordinal_distance` are explicit metadata for ORC-DPO weights and margins.

## Counts

- pairs: 429
- unique pair ids: 429
- all have human reason: True

## Risk Type Distribution

| risk_type | count |
|---|---:|
| `high_to_low_real_model_error` | 139 |
| `high_to_mid_real_model_error` | 131 |
| `low_to_high_real_model_error` | 130 |
| `low_to_mid_real_model_error` | 29 |

## Ordinal Distance Distribution

| distance | count |
|---:|---:|
| 1 | 108 |
| 2 | 123 |
| 3 | 106 |
| 4 | 92 |

## Guardrails

- Dev/test are used only for ID and question-key leakage checks.
- Test labels are not read.
- Human rationale is not in the user prompt.
