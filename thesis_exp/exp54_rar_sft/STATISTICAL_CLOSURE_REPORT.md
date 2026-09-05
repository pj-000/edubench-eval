# Exp54 统计收尾：冻结预测的 question/QA 聚类重算

## 1. 完成了什么

本次完成的是 RAR-SFT / Field-DPO 线的**统计重算**，不是新训练、不是
新一轮 test 推理，也不是整篇论文终稿验收。源分支锚点为
`a75f79d300fe9f7634247443950a268c6c836bfc`。HMSA 线不在本次范围内。

读取 42 份已有预测：SFT dev 12 份、5e-6 preference dev 9 份、正式 test
12 份、post-hoc mechanism test 9 份。495 项逐 seed 历史点指标对账一致。
没有删除原报告，没有重新挑 epoch/seed/arm，没有加载模型或使用 GPU。
本次确实重读已有 dev/test 元数据和预测，不能写成“未访问 test”。

原问题不是模型重新算错了分数，而是区间计算把同一个问题下的相关记录
当成独立样本。新分析保留原来的逐记录加权评价对象，但整组重采样问题。

| 数据 | 记录数 | question 数 | QA 数 | 低分记录数 | 含低分的 question 数 |
|---|---:|---:|---:|---:|---:|
| dev | 664 | 184 | 502 | 20 | 15 |
| test | 2,218 | 163 | 730 | 103 | 33 |

每种聚类做 10,000 次配对重采样；所有 arm 和三个固定 seed 共用抽样。
主分析按 question；QA 是敏感性分析；record 是诊断参照。Kendall/QWK
逐次从完整重采样混淆矩阵重算。主要端点均有 10,000 个有效重复。
少量 dev 重复没有抽到 Label-1/2 样本，其对应 Recall 被标为未定义而非零：
question 口径有效数分别为 9,980 / 9,999；所有口径最低有效数为 9,973，
均达到预设的 9,500 门槛。完整无效数随每个区间报告。

## 2. 主要结论：核心收益保留，附加模块的措辞进一步收紧

以下差值均为“新方法 − 对照”，MAE/L2H 为负表示改善。L2H 的差值与区间
已换算为**百分点**。区间为未校正的 question-cluster percentile 95% CI；
Holm p 来自预先固定的比较 family 和近似零假设中心化 bootstrap 检验。

| 对照 → 新方法 | MAE 差值 [95% CI] | MAE Holm p | L2H 差值（百分点）[95% CI] | L2H Holm p |
|---|---|---:|---|---:|
| R3 → P1 Field-DPO（正式 test） | −0.02450 [−0.03855, −0.01185] | 0.00350 | −16.18 [−21.46, −10.85] | 0.00060 |
| P1 → P2 offset（正式 test） | −0.00316 [−0.00768, +0.00061] | 0.15868 | −1.94 [−4.41, −0.29] | 0.15868 |
| P2 → P3 joint（正式 test） | −0.00631 [−0.01347, −0.00046] | 0.15868 | −3.24 [−6.36, −0.61] | 0.12319 |
| TOKENAVG → R3（post-hoc test） | −0.03051 [−0.04810, −0.01418] | 0.00240 | −4.53 [−11.20, +2.08] | 0.17188 |
| FULLSEQ → P1（post-hoc test） | −0.01833 [−0.03034, −0.00749] | 0.00690 | −11.65 [−16.87, −6.67] | 0.00060 |
| SYN → P1（post-hoc test） | −0.01788 [−0.03129, −0.00542] | 0.01340 | −11.00 [−16.67, −5.88] | 0.00150 |

另外将两组 test family 合并作 12 项 Holm 敏感性分析，P1−R3 的 MAE/L2H，
TOKENAVG−R3 的 MAE，以及 FULLSEQ/SYN−P1 的 MAE/L2H 仍通过 0.05 门槛。
这**不会**把事后机制实验升级成事前确认性实验。

合理解读：

- **P1 的核心结果保留。**在当前数据、模型、三 seed 和冻结推理协议下，
  评分误差及低分高估率改善仍有统计支持。不是只靠原来的 record bootstrap
  才能看到的现象。
- **P2 独立增量仍未确认。**不应把有利点估计写成 ordinal offset 已被证实有效。
- **P3 的组合点估计有利，但两个主要端点未通过新的 family Holm 门槛。**
  原报告对应端点的 Holm p 分别为 0.01680、0.01280；本次为 0.15868、
  0.12319。这不是“证明 P3 无效”。而且无论显著与否，P3 同时改变 score
  权重、pair 组成、有效 batch 和 FLOPs，不能独立归因到 rationale preference。
- 字段聚合、字段局部 DPO、实际错分来源的三个受控比较保留一定支持，
  但只是固定 recipe 下的 post-hoc 方法对照，不是通用算法定理或唯一因果机制。

未校正 percentile CI 不跨零、但多重比较校正后的 p 不通过，并不矛盾。
两者不是同一个判据；本次 p 还采用不同于旧报告的零假设中心化近似。
不能把 p 的全部变化仅归因于聚类，也不能把近似 bootstrap p 宣称为精确检验。

## 3. SFT dev：必须把“固定三个模型”与“任意训练 seed”分开

| dev 对照 → 方法 | MAE 差值 [question CI] | Holm p | 可以支持什么 |
|---|---|---:|---|
| S0 → R3 | −0.04669 [−0.06940, −0.02531] | 0.00060 | 固定三个选定模型的 dev 平均改善 |
| R1 → R3 | +0.00251 [−0.01070, +0.01553] | 1.00000 | 没有确认一致标签筛选优于 all-rater |
| R2 → R3 | −0.01305 [−0.02876, +0.00196] | 0.28857 | 语义对应的额外评分优势仍不确定 |

旧 S0→R3 MAE 区间为 [−0.10191, +0.00703]，包含零；新条件区间不含零。
这**不是模型或分数改变了**：旧脚本同时重采样 seed 和 record，新脚本固定
三个已训练模型，单独报告 seed 波动，再估计评价样本的不确定性。

S0 三 seed MAE 为 0.43524 / 0.39006 / 0.33735；R3 为
0.34187 / 0.33283 / 0.34789。配对差为 −0.09337 / −0.05723 / +0.01054，
不是每个 seed 都改善。S0 的 sample SD 是 0.04899，R3 是 0.00758。

因此新 CI 不能支持“所有随机种子稳定获胜”。这些 checkpoint 又是在同一 dev
上选择的，区间不含零也不等于独立 test 确认。正式 DPO test 没有 S0，不能
拿 P1→R3 的结果反向证明 RAR-SFT 相对 S0 的 test 优势。

## 4. 低分问题没有被完全解决

| 正式 test | seed42 | seed43 | seed44 |
|---|---|---|---|
| R3：低分→高分错误数 / 低分总数 | 61/103 | 48/103 | 86/103 |
| P1：低分→高分错误数 / 低分总数 | 51/103 | 36/103 | 58/103 |
| R3：Label-2 命中数 / Label-2 总数 | 0/47 | 0/47 | 0/47 |
| P1：Label-2 命中数 / Label-2 总数 | 1/47 | 0/47 | 1/47 |

P1 平均 L2H 从 63.11% 降至 46.93%，但 Label-2 Recall 仍仅约 1.42%。
可写“降低严重高估风险”，不能写“恢复低分等级识别”或“解决低分长尾”。
test 的 47 条 Label-2 记录只涉及 22 个问题，低分支持仍有限。

所有 42 份预测的严格解析率为 1.0，但 forced-close 仍存在；完整逐 seed
forced-close 数量、比例和混淆矩阵已发布。结构可解析不等于理由质量好。

## 5. 来源、验证与尚未关闭的边界

- 12 份 SFT dev 预测/metrics/protocol 核对已有公开 checkpoint-selection lock
  中的精确文件 SHA-256。
- 21 份 test 预测核对归档 completion receipt 中的预测/protocol SHA-256、
  arm/seed/行数与 split 哈希，并重现历史点指标。receipt 来自现有服务器
  档案；本次没有声称找到了全部 receipt 的事前公开独立哈希锚。
- 9 份 preference dev 核对 split/protocol、逐行身份/标签及公开历史点指标；
  本次未核验到独立的事前公开逐份 prediction hash。新报告记录当前读取的
  文件哈希，但新哈希不能倒过来证明过去从未改动。
- 机制分析的归档聚合报告/CSV 与已有公开 lock 中的四个哈希一致。
- 19 项 CPU 单元测试覆盖独立逐行指标/Kendall/QWK、非线性指标全样本重算、
  不等大聚类、配对/确定性、缺失子群、Holm、源身份/标签/分数类型篡改等。
- 不从 argmax 分数反推 NLL/Brier/RPS；本次输入未提供可用于这些指标的
  五类概率。没有为补指标启动推理。

统计重算不能修复数据划分或 test 暴露史，也不能证明私有人工标注的真实性。
下一步仍应完成历史 baseline 可比性表、test/post-hoc 时间线及正文主张收束。
这些是证据与写作工作，不是默认要求再训练。

## 6. 产物和复现

新结果目录：`thesis_exp/outputs/exp54_rar_sft/rar_v2/statistical_closure_v1/`。

- `cluster_results.json`：完整聚类支持、来源哈希、逐 seed/逐 arm/配对区间。
- `per_seed_metrics.csv`：42 个模型的原始计数、指标、混淆矩阵。
- `arm_uncertainty.csv`：指标均值、seed sample SD、条件聚类区间。
- `paired_contrasts.csv`：三个重采样口径、所有端点、family 和合并 Holm。
- `historical_comparison.csv`：保留历史区间、并列新 question 区间。
- `analysis_lock.json`：分析程序/方案/测试与生成结果文件的 SHA-256。

运行环境：Python / NumPy 的实际版本写入 JSON；只需 NumPy，测试另用 SciPy。
在仓库根目录运行（私有预测不随公开代码提交）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest thesis_exp.tests.test_exp54_statistical_closure -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m thesis_exp.exp54_rar_sft.statistical_closure \
  --private-input-dir tmp/statistical_closure_inputs \
  --output-dir tmp/statistical_closure_inputs/reproduction
```

输出目录必须不存在，避免覆盖旧结果。私有输入布局见程序中的固定 run
列表；公开报告不含个体 record/question/QA ID、人类理由或逐行预测。

本次工作支持把论文从“继续寻找训练改进”推进到“按现有证据写清贡献与边界”，
但不替代导师、学院对硕士论文的正式要求或全文验收。
