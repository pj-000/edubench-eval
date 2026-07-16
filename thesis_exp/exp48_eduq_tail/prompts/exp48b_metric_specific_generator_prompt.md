# Exp48B metric-specific rubric-grounded local-edit generator

You generate one qualification family from one train-only source blueprint. The blueprint includes one canonical evaluation metric and that metric's original benchmark rubric for levels 1-5. The rubric levels are authoritative. Do not replace them with a global rubric, do not invent generic E1/E2/S1 criteria, and do not use labels or answers from dev/test.

Create a materially new sibling question in the same language and educational setting, but change the substantive domain situation, entities, and requested task. Then compile this metric's exact rubric into three observable assertions:

- `D2`: a severe or critical defect whose presence specifically satisfies the original level-2 descriptor;
- `D3`: a moderate defect whose presence specifically satisfies the original level-3 descriptor but not D2;
- `H4`: the positive quality requirement needed for the original level-4 descriptor.

Each assertion must be specific to this metric. Factual accuracy, relevance, instruction following, reasoning rigor, clarity, personalization, and other metrics require different defects. Copy the corresponding original rubric descriptor verbatim into `rubric_quote`. Do not reduce all metrics to one shared semantic template.

Generate one fluent `base_answer` that realizes level 4 for this metric. Identify two different source spans that each occur exactly once in the base answer:

1. `score3_edit` replaces one span with a matched-length statement that introduces D3 while avoiding D2.
2. `score2_edit` replaces another span with a matched-length statement that introduces D2.

The caller, not you, constructs the final score-2 and score-3 answers by applying exactly one replacement to the same base answer. Requirements:

- source and replacement normalized-length ratio must be between 0.8 and 1.2;
- the two edits must target different unique spans;
- outside the replaced span, text must remain byte-for-byte identical;
- each replacement must be fluent and locally grammatical;
- score differences must come from rubric-relevant content, not brevity, broken grammar, score words, or obvious quality markers;
- `rubric_grounded_reason` must name the metric-specific defect and explain why the replacement matches the quoted level descriptor;
- do not output final constructed answers, intended criterion states, labels from the source, or any target-score phrase inside answer text.

Return exactly one JSON object conforming to `schemas/exp48b_metric_specific_edit_plan_schema.json`.
