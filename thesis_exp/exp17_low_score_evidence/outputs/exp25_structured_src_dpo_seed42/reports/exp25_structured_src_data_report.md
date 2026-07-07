# Exp25 Structured SRC-DPO Data Report

Exp25 builds same-schema reason/score consistency DPO pairs from train-only recovered human reasons.

## Dataset Counts

- R7H score_mismatch_only pairs: 429
- R7H structured mixed pairs: 1127
- dropped pairs: 0

## Negative Types

- high_protection_score_mismatch: 139
- low_failure_erasure_counterfactual: 130
- reason_mismatch_same_score: 429
- score_mismatch_same_reason: 429

## Guardrails

- Human reason is never placed in the user prompt.
- Dev/test are used only for sample_id and question_key leakage guards.
- Test labels are not read.
- Counterfactual rejected reasons are marked as counterfactual train-only negatives.
