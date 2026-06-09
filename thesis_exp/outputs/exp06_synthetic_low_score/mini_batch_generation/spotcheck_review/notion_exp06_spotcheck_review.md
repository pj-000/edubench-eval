# Exp6 Spot-check Review

## Status

Manual spot-check review is complete for the 17 filtered mini-batch synthetic samples.

## Counts

| item | count |
| --- | ---: |
| Total filtered | 17 |
| Accept as-is | 13 |
| Revise | 3 |
| Reject | 1 |
| Effective usable after revision | 16 |

## Readiness

| gate | decision |
| --- | --- |
| Direct full 384 generation | NO |
| Next 96-row batch generation | YES after prompt/filter hardening |
| Exp6 training | NO |

## Revision Queue

| synthetic_plan_id | action |
| --- | --- |
| mb004 | Revise error_type to `scenario_mismatch` or `rubric_violation`. |
| mb008 | Reject from low-score pool or relabel upward to `3_or_4`. |
| mb018 | Revise suggested label to `2`. |
| mb023 | Revise error_type to `superficial_fluency` or `rubric_violation`. |

## Prompt/filter Hardening

- Add label plausibility check.
- Add error_type alignment check.
- For target labels 1 and 2, require clear rubric-relevant failure.
- For target label 3, require boundary quality rather than clearly good quality.
- Avoid answers whose actual quality is higher than the target label.
- If a response partially satisfies the rubric too well, mark it for relabel or reject.

## Files

- `thesis_exp/outputs/exp06_synthetic_low_score/mini_batch_generation/spotcheck_review/manual_spotcheck_decisions.csv`
- `thesis_exp/outputs/exp06_synthetic_low_score/mini_batch_generation/spotcheck_review/prompt_hardening_recommendations.md`
- `thesis_exp/outputs/exp06_synthetic_low_score/mini_batch_generation/spotcheck_review/full_generation_readiness_after_manual_review.md`
