你正在为教育问答评价器生成仅供审计的 train-only synthetic 数据。

请根据给定教育任务生成一个自然、可信但有缺陷的回答。

约束：
- 使用与原始问题一致的语言：{language}。
- 目标评价维度：{metric_canonical}。
- 目标 5 分制伪标签：{target_label_5}。
- 错误类型：{error_type}。
- 根据 rubric 控制回答质量，使其大约符合目标分数。
- 回答要自然，不要显得刻意错误。
- 不要提到“故意错误”“synthetic”“低分”“实验生成”等元信息。
- 不要照抄原始回答。
- 不要修改原始问题。

原始问题：
{question}

评分细则：
{rubric_text}

只返回合法 JSON：
{{
  "answer_synthetic": "<自然但有缺陷的回答>",
  "target_label_5": {target_label_5},
  "error_type": "{error_type}",
  "brief_design_note": "<审计用简短说明，不进入训练文本>"
}}
