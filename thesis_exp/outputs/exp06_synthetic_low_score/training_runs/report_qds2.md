# Exp6 QD-S2 Human + Synthetic L1 Weighted Ordinal

This report compares QD-S2 against QD-B0, QD-B1, and QD-S1.
Synthetic rows are pseudo-label augmentation rows and are not human labels.

## Test Metrics

| run | MAE_label | QWK | Accuracy | low_to_high | Acc@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| QD-B0_human_only_ordinary_ordinal | 0.4019 | 0.5976 | 0.6936 | 0.5161 | 0.7873 |
| QD-B1_human_only_L1_weighted_ordinal | 0.4279 | 0.6012 | 0.6709 | 0.4516 | 0.7419 |
| QD-S1_human_plus_synthetic_ordinal | 0.4418 | 0.5544 | 0.6655 | 0.5484 | 0.7013 |
| QD-S2_human_plus_synthetic_L1 | 0.4566 | 0.5466 | 0.6555 | 0.4839 | 0.7029 |

## Class Weights

Class weights computed from QD-S2 train only:
- label 1: count=226, weight=3.000000
- label 2: count=221, weight=3.000000
- label 3: count=345, weight=2.150725
- label 4: count=1163, weight=0.638005
- label 5: count=1755, weight=0.500000

## Gate

- Can QD-S3 start? **REVIEW_REQUIRED**
- Model weights/checkpoints are written under `thesis_exp/artifacts/` and must not be committed.
