You are auditing whether the original human score is reliable.

Critical target rule:
- The blind teacher output should have scored only the evaluator output under
  `[EVALUATOR_OUTPUT_TO_SCORE]`.
- The `[CONTEXT_ONLY_ORIGINAL_TASK]` section is context only and may contain an
  embedded student answer. Do not audit by scoring that embedded answer.
- Always set `scored_target` to `evaluator_output_answer_field`.
- Set `target_confusion_detected=true` if the blind teacher output appears to
  grade the context/student answer instead of the evaluator output.

You will receive:
1. the original context, evaluator output, metric, rubric, and metadata,
2. the previous blind teacher output and its annotation id/hash,
3. the original human score for this train sample.

Return exactly one JSON object matching the provided audit schema. Do not wrap
the JSON in Markdown. Do not include chain-of-thought.

Important protocol rule:
- Do not copy or rewrite the blind object.
- Echo only `sample_id`, `blind_annotation_id`, `blind_annotation_hash`, and an
  `audit` object.
- The blind annotation is treated as fixed evidence for this audit pass. If the
  blind annotation seems questionable or target-confused, explain that inside
  `audit_reason` and choose conservative training use.

Audit rules:
- The teacher is an auditor, not a replacement gold-label source.
- Keep the blind teacher score conceptually separate from the original human
  score. Do not retroactively rationalize the original score.
- If the blind teacher score and original score differ by 0, use
  `score_agreement="exact"` unless another clear issue exists.
- If they differ by 1, usually use `score_agreement="adjacent"` and
  `label_quality="plausible_adjacent"`.
- If they differ by 2 or more, default to `score_agreement="conflict"`,
  `label_quality="suspected_conflict"`, `hard_conflict=true`, and
  `needs_human_review=true`, unless `audit_reason` clearly explains why the
  difference is acceptable.
- Use `label_noise_type` for rubric ambiguity, answer-key conflict,
  insufficient context, teacher strictness/leniency, target confusion, or
  annotator disagreement.
- `recommended_training_use` should be conservative:
  `high_weight` for reliable labels, `low_weight` for adjacent/plausible labels,
  `review_only` for conflicts or ambiguity, and `exclude` only for clearly
  unusable labels.
