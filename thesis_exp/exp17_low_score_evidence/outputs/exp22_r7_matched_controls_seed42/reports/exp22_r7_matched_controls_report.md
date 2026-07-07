# Exp22 R7 Matched Controls

Exp22 prepares train-only data for the next DPO experiments. It does not train a model.

## Outputs

- R7D source pairs: `429`
- R7E matched score-only pairs: `429`
- R7F consistency pairs: `856`
- R7F dropped candidates: `2`

## Alignment

- R7E exactly aligned with R7D: `True`
- R7D pairs not found in broad R7A pool: `0`

## R7F Negative Types

- reason_mismatch: `427`
- score_mismatch: `429`

## Leakage Audit

| dataset | pairs | pass | reason in prompt | dev overlap | test overlap |
|---|---:|---|---:|---:|---:|
| `edubench_r7e_matched_score_only_strict_real_dpo_train` | 429 | `True` | 0 | 0 | 0 |
| `edubench_r7f_score_reason_consistency_dpo_train` | 856 | `True` | 0 | 0 | 0 |

## Interpretation

- R7E is the strict score-only control required before comparing R7D against score-only DPO.
- R7F provides score-mismatch and reason-mismatch counterfactuals for SRC-DPO.
- Reason text is only present in assistant targets, never in user prompts.
- Dev/test splits are used only for ID and question-key leakage checks.
