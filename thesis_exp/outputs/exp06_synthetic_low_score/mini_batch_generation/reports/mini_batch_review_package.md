# Exp6-3 Mini-batch Review Package

- Can full 384 generation start? **NO**
- Can mini-batch generation start? **NO**
- Can Exp6 training start? **NO**
- Mode: **BLOCKED_NO_TRAIN_ONLY_SOURCE**
- Planned rows: **24**
- Generated raw rows: **0**
- Filter-passed rows: **0**
- Leakage status: **DRY_RUN_NO_GENERATED_ROWS**
- Spotcheck required: **YES**
- Labels are human-confirmed: **NO**
- Label status: **synthetic_design pseudo-label only**

Blockers:

- Full generation remains blocked until the 24-row mini-batch has real generated answers.
- If mode is `BLOCKED_NO_TRAIN_ONLY_SOURCE`, no API generation should start until source selection is
  fixed or manually approved.
- Training remains blocked until generated rows pass filtering, leakage, and manual spotcheck.
- Any dev/test leakage must block affected rows.

Next step:

Review
`thesis_exp/outputs/exp06_synthetic_low_score/mini_batch_generation/prompts/mini_batch_prompts.jsonl`
and the source-selection blocker.
If source selection is resolved and prompts are approved, run mini-batch generation with
`EXP6_RUN_GENERATION=1`, `GENERATION_MODEL`, and a valid key or endpoint, capped at 24 rows. Then
rerun normalization, filtering, leakage, spotcheck, sanity, and readability checks.
