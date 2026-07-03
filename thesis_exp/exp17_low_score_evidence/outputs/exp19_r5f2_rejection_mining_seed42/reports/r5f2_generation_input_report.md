# Exp19-R5F2 Generation Input QC

R5F2 expands train-only rejection mining. Human rationale is not included in prompts.

- train source: `thesis_exp/data/splits/question_seed42/train.jsonl`
- candidates selected: 111
- min answer chars for Pool A: 40
- include borderline y=3 pool: `False`

## Candidate Pools

- pool_a_strict_d1_like_low: 63
- pool_b_all_train_low: 48

## Failure Modes

- answer_key_or_reference_mismatch: 4
- format_violation: 3
- insufficient_evidence: 23
- missing_key_point: 25
- surface_fluent_but_hidden_defect: 25
- task_constraint_violation: 4
- unclear: 27

## Guardrails

- Test split is not read.
- Dev D1 annotations are not used as training labels.
- Human rationale-derived signals are used only for train-side candidate/target construction.
