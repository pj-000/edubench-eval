# Exp3 论文笔记：Rubric-aware Input Ablation

Exp3 将教育评分任务从普通文本分类扩展为 rubric-conditioned scoring。与 Exp2 相比，Exp3
不改变模型结构和 CE objective，而是系统比较 answer、question、metric、rubric 和 metadata
对人类一致性的影响。

A2 对应 Exp2 的 Q+A+metric baseline，因此可以直接复用 Exp2 正式结果。A3/A4 是核心设置，
用于检验显式 rubric 是否能降低低分样本被高估的问题，以及 metadata 是否进一步带来收益。

论文中需要谨慎说明：当前 rubric 来源于 split row field，但在 metric/language 组内重复，
应表述为 metric-level rubric description，而不是 sample-specific human annotation。
