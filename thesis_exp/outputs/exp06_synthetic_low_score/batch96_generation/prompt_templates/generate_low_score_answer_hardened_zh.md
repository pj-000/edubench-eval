你正在为教育问答评价器准备一个回答候选。

请根据原始任务写出自然、可信的中文回答。回答必须符合目标评价维度、目标标签和问题模式。

评价维度：{metric_canonical}
目标 1-5 分标签：{target_label_5}
问题模式：{error_type}

质量要求：
- 如果目标标签是 1，必须包含严重、清晰、与 rubric 直接相关的缺陷。
- 如果目标标签是 2，必须包含主要缺陷，即使表面流畅也不可靠。
- 如果目标标签是 3，必须是边界质量，有明显不足；不要生成明显好的答案。
- 如果回答明显高于目标标签，返回 `needs_revision=true`，并在 `revision_reason` 中说明原因。
- 回答的实际缺陷必须与 `error_type` 对齐；如果不对齐，返回 `needs_revision=true`。
- 不要使用“低分回答”“故意错误”“synthetic”“generated for an experiment”“合成”等痕迹短语。
- 不要照抄参考答案。
- 不要修改原始任务。

原始任务：
{question}

目标维度评分细则：
{rubric_text}

避免照抄的参考答案：
{source_answer}

只返回合法 JSON，字段必须完整：
{{
  "answer_synthetic": "<回答候选本身>",
  "target_label_5": {target_label_5},
  "error_type": "{error_type}",
  "rationale_for_label": "<审计说明：为什么目标标签可信>",
  "expected_failure_against_rubric": "<审计说明：指出清晰的 rubric 缺陷>",
  "label_plausibility_self_check": "pass or needs_revision",
  "error_type_alignment_self_check": "pass or needs_revision",
  "rubric_failure_visibility": "clear, boundary, weak, or needs_revision",
  "too_good_for_target_label": false,
  "needs_revision": false,
  "revision_reason": ""
}}
