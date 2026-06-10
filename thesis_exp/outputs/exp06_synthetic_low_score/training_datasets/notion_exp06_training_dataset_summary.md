# Exp6-13 Synthetic-Augmented Training Dataset Summary

- 数据集构建状态：**PASS**
- 是否可开始 Exp6 model training：**YES**
- 本阶段未调用 API、未生成新 synthetic、未训练模型。
- final synthetic low-score pool 作为 pseudo-label augmentation，只进入 train。

## Datasets

| Dataset | Train design | Dev/Test | Purpose |
| --- | --- | --- | --- |
| QD-S0_human_only | human train only | human only | baseline registration |
| QD-S1_human_plus_synthetic_ordinal | human + 384 synthetic | human only | ordinary ordinal |
| QD-S2_human_plus_synthetic_L1 | human + 384 synthetic | human only | L1 weighted ordinal |
| QD-S3_synthetic_pretrain_then_human_finetune | stage1 synthetic, stage2 human | human only | pretrain then finetune |

## Synthetic Label Distribution

| Label | Count |
| ---: | ---: |
| 1 | 168 |
| 2 | 168 |
| 3 | 48 |
