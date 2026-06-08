# Exp6-3 Mini-batch Review Package

- Can full 384 generation start? **YES**
- Can mini-batch generation start? **YES, after human prompt review**
- Can Exp6 training start? **NO**
- Mode: **GENERATED**
- Default generation split mode: **question_disjoint_formal**
- Planned rows: **24**
- Generated raw rows: **24**
- Filter-passed rows: **17**
- Leakage status: **PASS**
- Spotcheck required: **YES**
- Labels are human-confirmed: **NO**
- Label status: **synthetic_design pseudo-label only**

Blockers:

- Full generation remains blocked until the 24-row mini-batch has real generated answers.
- `paper_like_triple_pilot` is not allowed for training.
- `question_disjoint_formal` can proceed to 24-row mini-batch generation after human review.
- Training remains blocked until generated rows pass filtering, leakage, and manual spotcheck.
- Any dev/test leakage must block affected rows.

Next step:

Review
`thesis_exp/outputs/exp06_synthetic_low_score/mini_batch_generation/prompts/mini_batch_prompts.jsonl`.
If prompts are approved, run question-disjoint mini-batch generation with
`EXP6_RUN_GENERATION=1`, `GENERATION_MODEL`, and a valid key or endpoint, capped at 24 rows. Then
rerun normalization, filtering, leakage, spotcheck, sanity, and readability checks.
