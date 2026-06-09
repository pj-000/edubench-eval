# Exp6 Topup1 Curated Review

## 结果摘要

- 审阅模式：`auto_pass_plus_stratified_spotcheck_required`
- Topup1 filtered 总数：**152**
- 接受为候选可用：**152**
- revise_label：**0**
- revise_error_type：**0**
- reject：**0**
- usable_after_revision：**152**
- 标签分布：`1=73, 2=65, 3=14`
- 语言分布：`en=76, zh=76`
- 分层抽查子集：**36** 条

## 与 Batch96 合并后的累计

- Batch96 curated：**90**
- Topup1 curated：**152**
- 累计 usable：**242**
- 累计标签分布：`1=104, 2=109, 3=29`
- 距离最终目标剩余：`1=64, 2=59, 3=19`

## Gate 判断

- Topup2 是否可启动：**YES**
- 是否可以直接进入 full 384 generation：**NO**
- 是否可以开始训练：**NO**

说明：本轮未修改原始 `filtered_synthetic_candidates.jsonl`。Topup1 curated pool 采用 auto-pass + 分层 spotcheck 路径，后续仍需结合 Topup2 与总池继续审阅。
