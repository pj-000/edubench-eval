# Exp6 QD-S1 Synthetic Augmentation Summary

- Setting: question-disjoint `question_seed42`.
- Train: 3326 human + 384 final synthetic low-score pseudo-label rows.
- Dev/test: human-only.
- Loss: ordinary ordinal BCEWithLogitsLoss, no class weights, no asymmetric loss.

## Test Comparison

| Run | MAE_label | QWK | Accuracy | low_to_high | Acc@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| QD-B0_human_only_ordinary_ordinal | 0.4019 | 0.5976 | 0.6936 | 0.5161 | 0.7873 |
| QD-B1_human_only_L1_weighted_ordinal | 0.4279 | 0.6012 | 0.6709 | 0.4516 | 0.7419 |
| QD-S1_human_plus_synthetic_ordinal | 0.4418 | 0.5544 | 0.6655 | 0.5484 | 0.7013 |

- Can QD-S2 start: **REVIEW_REQUIRED**
