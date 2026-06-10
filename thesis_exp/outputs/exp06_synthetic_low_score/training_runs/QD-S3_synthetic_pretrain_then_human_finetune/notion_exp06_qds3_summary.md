# Exp6 QD-S3 Synthetic Pretrain -> Human Fine-tune Summary

- Setting: question-disjoint `question_seed42`.
- Stage 1: 384 final synthetic low-score pseudo-label rows.
- Stage 2: 3326 human-only train rows.
- Dev/test: human-only.
- Loss: ordinary ordinal BCEWithLogitsLoss, no class weights, no asymmetric penalty.

## Test Comparison

| Run | MAE_label | QWK | Accuracy | low_to_high | Acc@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| QD-B0_human_only_ordinary_ordinal | 0.4019 | 0.5976 | 0.6936 | 0.5161 | 0.7873 |
| QD-B1_human_only_L1_weighted_ordinal | 0.4279 | 0.6012 | 0.6709 | 0.4516 | 0.7419 |
| QD-S1_human_plus_synthetic_ordinal | 0.4418 | 0.5544 | 0.6655 | 0.5484 | 0.7013 |
| QD-S2_human_plus_synthetic_L1 | 0.4566 | 0.5466 | 0.6555 | 0.4839 | 0.7029 |
| QD-S3_synthetic_pretrain_then_human_finetune | 0.4569 | 0.5316 | 0.6247 | 0.6129 | 0.6607 |

- Exp6 training matrix complete: **YES**
- Can conclusion draft start: **REVIEW_REQUIRED**
