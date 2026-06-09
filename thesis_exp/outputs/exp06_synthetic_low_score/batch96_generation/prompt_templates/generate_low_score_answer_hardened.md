You are preparing one answer candidate for an education answer evaluator.

Write a natural, plausible answer to the original task in the requested language. The answer must
match the requested target label and issue pattern for the target metric.

Language: {language}
Metric: {metric_canonical}
Target label on the 1-5 scale: {target_label_5}
Issue pattern: {error_type}

Quality requirements:
- For target label 1, include a severe, clear, rubric-relevant failure.
- For target label 2, include a major defect that makes the answer unreliable even if it is
  superficially fluent.
- For target label 3, produce boundary quality with visible weaknesses; do not produce a clearly
  good answer.
- If the answer is clearly stronger than the requested target label, return `needs_revision=true`
  and explain why in `revision_reason`.
- Align the answer's actual weakness with `error_type`; if the weakness does not align, return
  `needs_revision=true`.
- Do not use artifact phrases such as "low-score answer", "intentionally wrong", "synthetic",
  "generated for an experiment", "低分回答", "故意错误", or "合成".
- Do not copy the reference answer.
- Do not modify the original task.

Original task:
{question}

Rubric for the target metric:
{rubric_text}

Reference answer to avoid copying:
{source_answer}

Return only valid JSON with exactly these fields:
{{
  "answer_synthetic": "<answer candidate only>",
  "target_label_5": {target_label_5},
  "error_type": "{error_type}",
  "rationale_for_label": "<audit note explaining why the target label is plausible>",
  "expected_failure_against_rubric": "<audit note naming the clear rubric weakness>",
  "label_plausibility_self_check": "pass or needs_revision",
  "error_type_alignment_self_check": "pass or needs_revision",
  "rubric_failure_visibility": "clear, boundary, weak, or needs_revision",
  "too_good_for_target_label": false,
  "needs_revision": false,
  "revision_reason": ""
}}
