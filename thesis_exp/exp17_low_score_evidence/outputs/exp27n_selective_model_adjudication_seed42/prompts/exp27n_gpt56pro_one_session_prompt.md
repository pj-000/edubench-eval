你现在作为 EduBench 教育评分数据的独立模型审阅者。

请一次性审阅附件 `exp27n_selective_adjudication_packets_54.jsonl`。文件共有且必须有 54 条 JSONL。

规则：

1. 只评价 `<EVALUATOR_OUTPUT_TO_SCORE>` 中的文本。
2. `<CONTEXT_ONLY_ORIGINAL_TASK>` 仅用于理解任务，不能直接回答原题或评分其中嵌套的学生答案。
3. 你看不到且不能猜测 human、Qwen、DeepSeek 或其他模型分数。
4. 严格依据当前样本的 metric 和 1–5 分 rubric。
5. 不因文本流畅、较长或格式漂亮自动给高分。
6. 检查事实或 rubric 不匹配、关键点缺失、证据不足、任务约束违反、score-reason 矛盾、学习者层次不匹配及评分对象混淆。
7. 题目、rubric 或答案键存在歧义时，使用 `failure_bucket="unclear"`、`confidence="low"` 和 `training_use="review_only"`，不要制造虚假的确定性。
8. `evaluator_output_evidence` 必须是待评分文本中的连续原文子串；缺失型问题可设为 `null` 并填写 `missing_evidence_reason`。
9. 不输出思维链，只给简短、可审计理由。
10. 一次审完全部 54 条，不要分批，也不要要求继续发送下一批。

每条输出严格使用：

```json
{
  "sample_id": "与输入完全一致",
  "target_scope_confirmed": true,
  "final_score_range": [1, 5],
  "final_score": 3,
  "failure_bucket": "no_failure | visible_failure | hidden_or_missing_failure | unclear",
  "major_failures": [],
  "rubric_evidence": "简短 rubric 依据",
  "evaluator_output_evidence": "待评分文本连续原文或 null",
  "missing_evidence_reason": null,
  "score_cap": null,
  "confidence": "high | medium | low",
  "training_use": "resolved_model_silver | review_only",
  "review_status": "completed"
}
```

约束：

- `final_score` 必须位于 `final_score_range` 内。
- `failure_bucket=no_failure` 时，`major_failures` 必须为空数组。
- `failure_bucket=unclear` 或 `confidence=low` 时，`training_use` 必须为 `review_only`。
- 不遗漏、增加或重复任何 `sample_id`。
- 最终恰好输出 54 行 JSONL，每行一个 JSON 对象。
- 不使用 Markdown 代码块，不输出标题、总结、表格、序号或说明。

这批结果属于 single-strong-model selective adjudication，只能称为 model-reviewed silver，不能称为 human gold 或 expert annotation。
