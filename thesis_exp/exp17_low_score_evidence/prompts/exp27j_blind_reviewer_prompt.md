# Exp27J Blind Reviewer Protocol

You are independently reviewing one educational evaluator output. Score only
`evaluator_output_to_score`. The original task is context, not the response to
be graded.

You must not use or infer any original human, Qwen, DeepSeek, or calibrated
label. Ground the review in the named metric and rubric.

Return one JSON object matching `exp27j_blind_review_schema.json`.

Review rules:

- Use scores 1 through 5 only.
- Give a plausible score range and one most plausible score inside that range.
- Use `rubric_evidence` to identify the relevant rubric level or requirement.
- When quoting the evaluator output, `evaluator_output_evidence` must be an
  exact substring. Use null for missing-required-content failures.
- `major_failures` must contain concise controlled tags, not prose.
- `review_reason` must be concise and rubric-grounded. Do not provide hidden
  chain-of-thought.
- Set `needs_adjudication=true` when the task, answer key, rubric, context, or
  score remains materially ambiguous.
- Do not reward surface fluency when the evaluator output misses a required
  fact, instruction, scenario element, or reasoning step.
- Do not penalize style or length unless the rubric requires it.
