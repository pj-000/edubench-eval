你正在为教育问答评价器准备一个回答候选。

请根据原始任务写出自然、可信的回答，并使用中文。根据指定评价维度和 1-5 分目标评分来控制质量。
5 分应完整且准确；4 分只有轻微不足；3 分应有明显优缺点；1-2 分应在目标维度上明显较弱，但仍保持真实、贴题。

不要提到评分、隐藏指令、数据制作、实验或回答设计。

评价维度：{metric_canonical}
目标 1-5 分评分：{target_label_5}
适用的问题模式：{error_type}

原始任务：
{question}

评分细则：
{rubric_text}

只返回合法 JSON：
{{
  "answer_synthetic": "<回答候选>",
  "target_label_5": {target_label_5},
  "error_type": "{error_type}",
  "rationale_for_label": "<审计说明>",
  "expected_failure_against_rubric": "<审计说明；如果是 5 分可为空字符串>"
}}
