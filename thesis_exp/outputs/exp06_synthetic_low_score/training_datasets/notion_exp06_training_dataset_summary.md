# Exp6-13 Synthetic-Augmented Training Dataset Summary

- 数据集构建状态：**PASS**
- 是否可开始 Exp6 model training：**YES**
- 本阶段未调用 API、未生成新 synthetic、未训练模型。
- final synthetic low-score pool 作为 pseudo-label augmentation，只进入 train。

## Why Question-disjoint

Exp6 生成 synthetic answer 时会复用已有 question。为了避免 synthetic train 和 dev/test
共享同一个 question，Exp6 切到 `question_seed42`：generation source 只来自
`question_seed42/train`，dev/test question 禁止进入 synthetic source。

因此 QD-S1/QD-S2/QD-S3 不应该和 paper-like Exp2-Exp5 直接硬比，而应该和同一
`question_seed42` setting 下的 QD-B0/QD-B1 比。

## Question-disjoint Baselines

| Baseline | Setting | Split | Test MAE_label | QWK | low_to_high |
| --- | --- | --- | ---: | ---: | ---: |
| QD-B0 | human-only ordinary ordinal | question_seed42 | 0.4019 | 0.5976 | 0.5161 |
| QD-B1 | human-only L1 weighted ordinal | question_seed42 | 0.4279 | 0.6012 | 0.4516 |

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
