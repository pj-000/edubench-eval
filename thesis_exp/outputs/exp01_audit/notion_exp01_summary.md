# Exp1 自动 Judge 与人类评分一致性审计

## 1. 实验目的

审计自动评估器与人类评分的一致性，重点定位低分盲区、系统性高估，以及 metric、scenario、subject 等分层差异。

## 2. 输入数据

- 主数据集：`thesis_exp/data/processed/edubench_scoring_all.jsonl`
- 数据集名称：`edubench_audit_human_scored_subset`
- 主测试集：`thesis_exp/data/splits/paper_like_triple_seed42/test.jsonl`
- test rows：2218
- 不训练模型，不调用 API，不使用 synthetic/sample 数据。

## 3. 输出结果

- 对齐预测：`thesis_exp/outputs/exp01_audit/predictions_aligned.jsonl`
- 主指标表：`thesis_exp/outputs/exp01_audit/tables/evaluator_metrics.csv`
- 低分审计表：`thesis_exp/outputs/exp01_audit/tables/low_score_metrics.csv`
- 报告：`thesis_exp/outputs/exp01_audit/report.md`

## 4. 核心指标

| evaluator | n_valid | MAE | Signed Bias | Exact Match | Kendall tau |
| --- | --- | --- | --- | --- | --- |
| EduBenchEvaluator | 2218 | 0.4358280733393447 | 0.25278028253682 | 0.7240757439134355 | 0.5051636452404249 |
| GPT-4o | 2192 | 0.6027980535279807 | 0.4802311435523114 | 0.5752737226277372 | 0.2767656507351872 |
| DeepSeek-R1 | 2218 | 0.5952810339645327 | 0.3379921851517884 | 0.584761045987376 | 0.3154341848109602 |
| DeepSeek-V3 | 2218 | 0.5807033363390441 | 0.4601743312293357 | 0.6032461677186655 | 0.3243146328338703 |
| QwQ-plus | 2196 | 0.5980570734669096 | 0.4074074074074075 | 0.604735883424408 | 0.2992004212761405 |

## 5. 低分盲区结论

| evaluator | n_valid_low | low_exact_match | low_recall | low_signed_bias | low_to_high_rate | mean_pred_low |
| --- | --- | --- | --- | --- | --- | --- |
| EduBenchEvaluator | 103 | 0.3495145631067961 | 0.3883495145631068 | 1.812297734627832 | 0.5339805825242718 | 3.320388349514563 |
| GPT-4o | 103 | 0.0 | 0.0 | 3.0550161812297736 | 0.912621359223301 | 4.563106796116505 |
| DeepSeek-R1 | 103 | 0.0970873786407767 | 0.0970873786407767 | 2.7734627831715213 | 0.7669902912621359 | 4.281553398058253 |
| DeepSeek-V3 | 103 | 0.0485436893203883 | 0.0485436893203883 | 2.9385113268608416 | 0.912621359223301 | 4.446601941747573 |
| QwQ-plus | 101 | 0.0693069306930693 | 0.0792079207920792 | 3.016501650165017 | 0.8514851485148515 | 4.524752475247524 |

低分段 label<=2 的识别明显弱于高分段，自动 judge 倾向于把低质量回答判到更高等级。

## 6. 与小论文结果关系

PDF 主要趋势复现状态：**YES**。本实验不使用 reference value 修正当前结果，只作为趋势对照。

## 7. 对后续实验的影响

Exp2 需要建立 CE baseline；Exp3 关注 rubric-aware 输入；Exp5 关注低分敏感 loss；Exp6 才能进入 synthetic low-score
augmentation；Exp7 需要做 calibration。

## 8. 当前风险和限制

- 只使用已有 judge predictions。
- missing evaluator 不补全、不猜测。
- subject 分层来自 local enriched metadata，只作为审计 metadata。
- 后续不能只看 overall accuracy。
