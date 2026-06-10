# Exp6 QD-S3 Synthetic Pretrain then Human Fine-tune

This report compares QD-S3 against QD-B0, QD-B1, QD-S1, and QD-S2.
Stage 1 uses synthetic low-score pseudo labels only; stage 2 returns to human-only question_seed42
training.
Synthetic rows are never treated as human labels.

## Test Metrics

| run | MAE_label | QWK | Accuracy | low_to_high | Acc@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| QD-B0_human_only_ordinary_ordinal | 0.4019 | 0.5976 | 0.6936 | 0.5161 | 0.7873 |
| QD-B1_human_only_L1_weighted_ordinal | 0.4279 | 0.6012 | 0.6709 | 0.4516 | 0.7419 |
| QD-S1_human_plus_synthetic_ordinal | 0.4418 | 0.5544 | 0.6655 | 0.5484 | 0.7013 |
| QD-S2_human_plus_synthetic_L1 | 0.4566 | 0.5466 | 0.6555 | 0.4839 | 0.7029 |
| QD-S3_synthetic_pretrain_then_human_finetune | 0.4569 | 0.5316 | 0.6247 | 0.6129 | 0.6607 |

## Gate

- Exp6 training matrix complete? **YES**
- Can Exp6 training conclusion draft start? **REVIEW_REQUIRED**
- Model weights/checkpoints are written under `thesis_exp/artifacts/` and must not be committed.
