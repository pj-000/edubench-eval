You are generating audit-only synthetic training data for an education answer evaluator.

Generate a plausible but flawed student/assistant answer for the given educational task.

Constraints:
- Use the same language as the original question: {language}.
- Target metric: {metric_canonical}.
- Target 5-point pseudo-label: {target_label_5}.
- Error type: {error_type}.
- Use the rubric to make the answer score around the target label for this metric.
- Keep the answer natural and realistic.
- Do not mention that the answer is intentionally flawed, synthetic, low-score, or generated for an experiment.
- Do not copy the original answer if one is visible in the task.
- Do not change the original question.

Original question:
{question}

Rubric:
{rubric_text}

Return only valid JSON with this schema:
{{
  "answer_synthetic": "<natural flawed answer>",
  "target_label_5": {target_label_5},
  "error_type": "{error_type}",
  "brief_design_note": "<short note for audit logs, not part of training text>"
}}
