你现在在 pj-000/edubench-eval 仓库中继续 Exp27G：teacher-audited 361-case expansion。

背景：
Exp27F 已完成 Exp27E top40 冲突仲裁 pilot。结论是：原始人类标签和 teacher 标签都存在冲突，不能直接扩大到全量 3326；但可以进入 361-case
controlled expansion。Exp27F 不是最终 gold，只是用于确定扩展协议。

目标：
只在 train split 中构造 361 条 teacher-audited annotation packets，并运行 API 标注或生成可运行脚本。不要读取 dev/test
labels，不训练模型。

要求：
1. 输入：
   - thesis_exp/data/splits/question_seed42/train.jsonl
   - Exp27D/Exp27E/Exp27F 的轻量结果
2. 输出目录：
   - thesis_exp/exp17_low_score_evidence/outputs/exp27g_teacher_audited_361_seed42/
3. 需要输出：
   - packets/exp27g_361_teacher_packets.jsonl
   - tables/exp27g_sampling_distribution.csv
   - tables/exp27g_leakage_audit.csv
   - reports/exp27g_prepare_report.md
   - decision/exp27g_prepare_decision.json
4. 抽样策略：
   - 覆盖 train 中低分、隐藏失败、teacher-human 冲突、高分保护控制样本。
   - 不允许 dev/test sample_id 或 question_key 泄漏。
   - 对 conflict-prone/risk-prone 样本保留 second-teacher 标注标记。
5. 如果运行 API：
   - 不提交 raw API outputs/logs。
   - 只提交 parsed lightweight CSV/MD/JSON。
6. 验证：
   - python -m py_compile 新增脚本
   - 运行 prepare 脚本
   - 检查 leakage_audit 全 0

最终回复请汇报：
- 构造了多少 361 packets；
- 各风险桶/分数段/语言/metric 分布；
- 是否有泄漏；
- 是否建议开始真实 API 标注；
- 下一步 Codex 命令。
