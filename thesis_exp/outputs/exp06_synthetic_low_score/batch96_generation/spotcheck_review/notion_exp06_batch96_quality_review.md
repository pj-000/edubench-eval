# Exp6-7 Batch96 全量人工质检表

## 本次任务

基于已完成的 Batch96 filtered synthetic samples，生成 91 条全量人工审查决策表与质量汇总模板。本步骤未调用 API、未生成新 synthetic、未训练模型，也未修改原始 `filtered_synthetic_candidates.jsonl`。

## 输入状态

- Filtered samples: **91**
- Leakage: **PASS**
- 当前标签分布: **label 1=37, label 2=39, label 3=15**
- 当前语言分布: **en=47, zh=44**
- Metric coverage: **12**
- Error type coverage: **7**

## 生成文件

- `spotcheck_review/batch96_manual_spotcheck_decisions.csv`
- `spotcheck_review/prefilled_quality_flags.csv`
- `spotcheck_review/batch96_spotcheck_summary_template.csv`
- `spotcheck_review/batch96_full_generation_readiness.md`
- `spotcheck_review/batch96_topup_generation_plan.csv`
- `spotcheck_review/notion_exp06_batch96_quality_review.md`

## 人工审查说明

`batch96_manual_spotcheck_decisions.csv` 每条 filtered sample 一行，共 **91** 行。人工 reviewer 需要填写质量字段与 decision，decision 只能使用：`accept`、`revise_label`、`revise_error_type`、`reject`。

`prefilled_quality_flags.csv` 是启发式风险预填，不是人工结论。当前启发式标记：

- needs_human_attention: **28**
- possible_label_too_low: **6**
- possible_error_type_mismatch: **10**
- possible_artifact: **0**

## Readiness

由于人工字段尚未填写：

- Full/top-up generation: **NO, pending full spot-check**
- Top-up generation: **review 后 usable_after_revision >= 80% 且 leakage PASS 才可 YES**
- Exp6 training: **NO，直到 full filtered pool complete**

## Top-up 计划

当前 filtered label distribution: **1=37, 2=39, 3=15**。
最终目标: **1=168, 2=168, 3=48**。
剩余目标: **1=131, 2=129, 3=33**。

计划先做两个约 160 raw 的 top-up batch；第三批仅在前两批过滤、泄漏检查、人工质检后仍不足时启动。
