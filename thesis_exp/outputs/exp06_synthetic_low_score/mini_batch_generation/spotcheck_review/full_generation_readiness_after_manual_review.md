# Exp6 Full-generation Readiness After Manual Review

## Review Summary

- Total filtered: **17**
- Accept as-is count: **13**
- Revise count: **3**
- Reject count: **1**
- Effective usable after revision count: **16**

## Decision

- Direct full 384 generation can start: **NO**
- Next 96-row batch generation can start: **YES, after prompt/filter hardening**
- Exp6 training can start: **NO**

## Rationale

The mini-batch is broadly usable, but manual review found one sample that is too strong for the
low-score pool and three samples requiring correction before reuse. This is enough evidence to move
from a 24-row pilot to a larger 96-row batch, but not enough to start direct full 384-row generation
or any Exp6 training run.

Before the next batch, the generation prompt and filtering checklist should enforce label
plausibility, error_type alignment, and stricter low-score failure visibility.

## Required Revisions Before Reuse

| synthetic_plan_id | decision | required action |
| --- | --- | --- |
| mb004 | revise_error_type | Change error_type to `scenario_mismatch` or `rubric_violation`. |
| mb008 | reject_from_low_score_pool or relabel_upward | Remove from low-score pool, or relabel upward to `3_or_4` outside low-score augmentation. |
| mb018 | revise_label | Change suggested label to `2`. |
| mb023 | revise_error_type | Change error_type to `superficial_fluency` or `rubric_violation`. |

## Artifact References

- Filtered samples:
  `thesis_exp/outputs/exp06_synthetic_low_score/mini_batch_generation/filtered/filtered_synthetic_candidates.jsonl`
- Manual decisions:
  `thesis_exp/outputs/exp06_synthetic_low_score/mini_batch_generation/spotcheck_review/manual_spotcheck_decisions.csv`
- Prompt hardening:
  `thesis_exp/outputs/exp06_synthetic_low_score/mini_batch_generation/spotcheck_review/prompt_hardening_recommendations.md`
