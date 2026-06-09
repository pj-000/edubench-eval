# Exp6-8 Batch96 人工审阅决策应用与 Curated Pool

## 本次结论

本步骤根据 Batch96 的人工审阅规则与启发式风险 flags，填充 91 条 filtered synthetic samples 的人工决策，并生成 curated usable pool、revision log、rejected archive 与 top-up readiness。未调用 API、未生成新 synthetic、未训练模型，也未修改原始 `filtered_synthetic_candidates.jsonl`。

## 决策统计

- Total filtered: **91**
- Accept: **75**
- Revise label: **5**
- Revise error type: **10**
- Reject: **1**
- Usable after revision: **90**
- Usable rate after revision: **0.9890**

## Curated 分布

- Label 1 usable: **31**
- Label 2 usable: **44**
- Label 3 usable: **15**
- EN usable: **46**
- ZH usable: **44**

## 输出文件

- `curated_review/batch96_manual_spotcheck_decisions_filled.csv`
- `curated_review/curated_batch96_synthetic_candidates.jsonl`
- `curated_review/rejected_or_relabel_upward_batch96.csv`
- `curated_review/batch96_revision_log.csv`
- `curated_review/batch96_curated_summary.csv`
- `curated_review/batch96_topup_readiness_after_review.md`
- `curated_review/batch96_topup_plan_after_review.csv`
- `curated_review/sanity_check_curated_batch96.md`
- `curated_review/sanity_check_curated_batch96.csv`

## Gate

- Top-up generation can start: **YES**
- Direct full 384 generation: **NO**
- Exp6 training: **NO**
- Next step: 按 label deficit 做 top-up generation，并继续 question-disjoint、leakage check、manual review。
