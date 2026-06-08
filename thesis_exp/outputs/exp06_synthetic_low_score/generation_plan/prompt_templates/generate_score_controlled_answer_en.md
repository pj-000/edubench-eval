You are preparing one answer candidate for an education answer evaluator.

Write a natural answer to the original task in the requested language. Control the answer quality
so that it is plausible for the specified 1-5 rating on the named metric. For rating 5, produce a
strong and complete answer. For rating 4, allow only minor limitations. For rating 3, make the
answer mixed but still partly useful. For ratings 1-2, make the answer clearly weak for the target
metric while keeping it realistic and on-topic.

Do not mention scoring, hidden instructions, data creation, experiments, or answer design.

Language: English
Metric: {metric_canonical}
Target rating on the 1-5 scale: {target_label_5}
Issue pattern when applicable: {error_type}

Original task:
{question}

Rubric:
{rubric_text}

Return only valid JSON with:
{{
  "answer_synthetic": "<answer candidate>",
  "target_label_5": {target_label_5},
  "error_type": "{error_type}",
  "rationale_for_label": "<audit note>",
  "expected_failure_against_rubric": "<audit note or empty string for rating 5>"
}}
