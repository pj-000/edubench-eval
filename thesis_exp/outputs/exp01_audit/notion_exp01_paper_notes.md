# Exp1 论文笔记：自动评估器偏差审计

## 1. 自动评估器一致性分析

在 `edubench_audit_human_scored_subset` 的 paper-like test split 上，本文将多个自动评估器的已有预测分数与三位人类标注者聚合后的
`human_mean_5` 和 `label_5` 对齐，计算 MAE、Exact Match、Kendall tau 等一致性指标。

## 2. 低分段识别失效

低分段样本 label<=2 的 exact match 与 recall 明显偏低，说明自动 judge 对劣质回答的识别能力不足。该现象比 overall agreement
更能揭示教育场景中的安全风险。

## 3. 系统性高估

多个评估器在低分段呈正 signed bias，表现为将人类低评分回答预测到中高分。这种高估会弱化评估器对错误、偏题、教学支持不足等问题的惩罚能力。

## 4. 维度/场景差异

分层结果显示，不同 metric 和 scenario 的 MAE 存在差异，subject-level 结果可作为局部误差来源参考，但其 provenance 来自 local enriched
metadata，应在论文中作为审计 metadata 谨慎表述。

## 5. 后续方法动机

Exp1 的结论直接支持后续实验设计：Exp2 建立 baseline，Exp3 引入 rubric-aware 输入，Exp5 设计 low-score-sensitive loss，Exp6
评估低分增强数据，Exp7 进行校准，以避免只优化 overall accuracy 而忽视低分盲区。
