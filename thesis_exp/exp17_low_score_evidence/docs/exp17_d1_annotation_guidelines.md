# Exp17-D1 Annotation Guidelines

Exp17-D1 audits whether label-2 high-prediction cases are rubric-conditioned
hidden failures: answers that look fluent or complete on the surface but fail a
rubric clause, task constraint, format requirement, factual requirement, or
contextual expectation that human raters used.

This is a diagnostic workflow. Do not read test data, do not train on these dev
annotations directly, and do not treat model/LLM over-scoring as a primary
failure mode.

## Annotation Order

1. Read `question`, `metric`, and `rubric` first.
2. Read the `answer`.
3. Try to identify the exact rubric clause or task constraint that explains the
   human low score.
4. Compare matched controls only as supporting context.
5. Fill the manual fields with the controlled labels below.

## Primary Failure Modes

- `format_violation`: the answer violates an explicit output format, such as
  malformed JSON or extra non-JSON text when JSON only is required.
- `task_constraint_violation`: the answer misses an instruction or task
  constraint that is not only formatting.
- `factual_or_rubric_mismatch`: the answer conflicts with a factual requirement
  or with the rubric standard.
- `surface_fluent_but_hidden_defect`: the answer is fluent and plausible, but a
  hidden rubric/task failure explains the human low score.
- `missing_key_point`: the answer omits a key expected point.
- `insufficient_evidence`: the answer gives too little evidence, explanation, or
  justification.
- `possible_label_conflict`: the human low label is hard to explain even after
  reading the rubric and task context.
- `other`: use only when none of the above fits; explain in `defect_notes_manual`.
- `unclear`: use when the case needs another review.

`over_scoring` is not a primary failure mode. It describes that a model or LLM
judge assigned a high score, not the evidence inside the answer. If needed, use
`llm_or_model_over_scoring_pattern_manual`.

## Rubric Link Levels

- `explicit_rubric_clause`: the failure is directly tied to a written rubric
  clause.
- `implicit_task_constraint`: the failure is tied to a task instruction or
  output constraint.
- `inferred_from_context`: the failure is plausible from question context, but
  not explicit in the rubric.
- `not_rubric_linked`: no rubric or task link is found.
- `unclear`: needs more review.

## Trainability Labels

- `strong_train_signal`: clear, rubric-linked, likely reusable.
- `weak_train_signal`: useful but noisy or partially inferred.
- `format_auxiliary_signal`: useful for format/task auxiliary supervision.
- `pairwise_only`: better used in case-control or hard-negative pairs than as a
  direct evidence-positive label.
- `downweight_or_exclude`: likely conflict/noise; avoid using as positive
  training signal.
- `review_only`: keep for qualitative analysis only.
- `unclear`: undecided.

## Recommended Training Use

- `evidence_positive`: use as hidden-failure evidence-positive candidate.
- `format_auxiliary`: use as auxiliary format/task signal.
- `pairwise_low`: use as low side in matched hard-negative separation.
- `downweight`: keep but reduce weight.
- `exclude`: exclude from training signal construction.
- `review_only`: use only in the report.
- `unclear`: needs review.

## Field Rules

- `is_surface_fluent_manual`: `1` if the answer looks fluent/complete at first
  glance.
- `is_hidden_failure_manual`: `1` if the low-score reason is not obvious from
  surface fluency alone but becomes visible through rubric/task context.
- `is_format_or_task_constraint_manual`: `1` for format or instruction failures.
- `possible_label_conflict_manual`: `1` if you cannot explain the human low
  label with the question, answer, rubric, or task constraint.
- `rubric_clause_manual`: quote or summarize the exact rubric/task clause.
- `evidence_span_manual`: quote the answer span that supports the defect.
- `confidence_manual`: integer 1-5.

If the answer only looks over-scored by a model, do not mark over-scoring as the
primary failure. Find the answer-side failure evidence, or mark
`possible_label_conflict_manual=1`.
