# 可直接发送给 GPT-6 Pro 的统计收尾审阅 prompt

请独立审阅我的硕士论文 Exp54（RAR-SFT / Field-DPO）统计修订。你上一轮的
核心要求是：从冻结预测关闭 question/QA 相关性、固定 seed 条件区间、
比较 family、多重校正和结论边界，而不是继续增加训练。

仓库：https://github.com/pj-000/edubench-eval

本次唯一审阅目标提交：
https://github.com/pj-000/edubench-eval/commit/29da9c750197406c5882550908eca0181cad6cae

精确文件树：
https://github.com/pj-000/edubench-eval/tree/29da9c750197406c5882550908eca0181cad6cae

不要把 PR #2 的缓存页面或别的分支最新文件替代这个提交。请先报告实际可读
文件和未读证据；不要把汇总中的 PASS 当作独立验证。若原始私有预测不可读，
请明确区分“实现与公开结果一致性审阅”和“独立重跑私有数据”，不要声称后者。

## 请优先阅读

源码目录 `thesis_exp/exp54_rar_sft/`：

1. `STATISTICAL_CLOSURE_PROTOCOL.md`：重算前固定的分析口径。
2. `statistical_closure.py`：CPU-only 实现。
3. `STATISTICAL_CLOSURE_REPORT.md`：中文解释及主张边界。
4. 原 `collect_dev_results.py`、`collect_sorc_dpo_dev_results.py`、
   `collect_sorc_dpo_test_results.py`、`collect_mechanism_control_test_results.py`。

测试：`thesis_exp/tests/test_exp54_statistical_closure.py`。

公开产物目录：
`thesis_exp/outputs/exp54_rar_sft/rar_v2/statistical_closure_v1/`：
`cluster_results.json`、`per_seed_metrics.csv`、`arm_uncertainty.csv`、
`paired_contrasts.csv`、`historical_comparison.csv`、`analysis_lock.json`。

## 本轮实际做了什么

未重新训练、未加载模型、未重新生成 dev/test。只重读 42 份已存在预测：
SFT dev 12、preference dev 9、正式 test 12、事后机制 test 9。
495 项逐 seed 历史点指标对账一致；原报告没有改写。

dev 664 条、184 个 question、502 个 QA；test 2,218 条、163 个 question、
730 个 QA。主重采样单位为 question，QA 敏感性，record 固定-seed 诊断；
每种 10,000 次。每次保留抽中 cluster 的全部记录和重复拷贝，仍以记录为
权重计算指标，不改成平均每个 cluster 的指标。

三个训练 seed 固定，逐 seed 计算完整指标再平均。区间只估计条件于这三个
模型的样本不确定性；另报 seed sample SD 和配对 seed 差。Kendall/QWK
从每次完整重采样的混淆矩阵重算。缺失低分子群的 Recall 为未定义，不填零。

CI 是 percentile 95%。近似双侧 p 使用
`(1 + count(abs(delta_boot - delta_observed) >= abs(delta_observed))) /
(1 + valid_replicates)`；不是精确随机化检验。每个预定 family 对 MAE/L2H
作 Holm，另报合并两组 test family 的 12 项 Holm 敏感性。请独立判断这个
统计选择是否合理，而不要因实现满足文字方案就直接认定方法无问题。

## 请回答的关键问题

1. 实现是否真正保持 cluster 整组、跨 arm/seed 配对、逐记录加权和固定 seed
   estimand？非线性指标、缺失子群、样本标准差和 Holm 是否正确？是否存在
   会改变结果的代码错误或统计假设缺口？
2. 新的近似 p 与 percentile CI 是否得到了准确说明？如果你认为当前 p
   不合适，请给出适用于现有冻结预测的替代及依据，并明确这是否会实质改变
   论文结论，而不是只因有另一种方法就要求重做。
3. S0→R3 原 CI 跨零，而新条件 CI 不跨零。报告是否充分解释这主要涉及
   seed 不确定性定义变化，且同 dev 选 checkpoint 后的区间并非独立确认？
   是否仍有“统计修正后宣布普遍胜利”的风险？
4. P1−R3 的 MAE/L2H 在 question 聚类及 Holm 后仍有支持；P2 未确认；P3
   原显著性减弱且本来就存在组合干预混杂。现在的措辞是否与证据一致？
5. TOKENAVG/FULLSEQ/SYN 的结果是否被正确限制为同一旧 test 上的事后
   对照，而没有升级成通用算法创新、独立确认性机制或跨域泛化结论？
6. 数据来源核验分级是否诚实？12 份 SFT 有既有公开 prediction hash；
   21 份 test 核对服务器归档 receipt 和历史指标；9 份 preference dev
   未检查到事前公开逐份 prediction hash。新文件哈希不证明历史不可变。
   在此明确限制下，当前统计结论能够承担多强的论文主张？

请最终给出：

- 本次统计收尾是否完成，或还存在什么会实质影响结论的阻断项。
- RAR-SFT、Field-DPO、offset、joint rationale、三个事后机制对照各自
  “能写什么 / 不能写什么”。
- 下一步是否可以只完成 baseline 可比性表、test 暴露时间线和论文写作；
  如认为必须新增计算，说明哪条科学结论缺少何种证据、最低成本如何补。

请允许反驳方案本身，也不要把“未显著”自动写成“方法没有作用”。若有问题，
请给出具体文件/函数、反例和最小修正；若主要结论已稳固，不必扩展新的模型
训练或重新设计整篇论文。硕士毕业充分性仍应附带导师/学院标准和终稿验收条件。
