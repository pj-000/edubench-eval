# Exp9 QD-PR1 Pairwise Ordinal Review Package

Formal training status: `pending_formal_training`
Training executed by setup stage: `no`.
API called: `no`.
Synthetic generated: `no`.

## Method

QD-PR1 adds risk-aware ordinal preference pairs on top of QD-B1-style weighted ordinal BCE.
Pairs are built from QD-S0 human-only train rows; dev pairs are diagnostic only.

## Test Comparison

| run | status | MAE_label | QWK | Accuracy | low_to_high | Acc@5 | monotonic_violation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QD-B0_human_only_ordinary_ordinal | completed | 0.4019 | 0.5976 | 0.6936 | 0.5161 | 0.7873 | 0.1985 |
| QD-B1_human_only_L1_weighted_ordinal | completed | 0.4279 | 0.6012 | 0.6709 | 0.4516 | 0.7419 | 0.3119 |
| QD-R1_CORAL_human_only | completed | 0.4467 | 0.5272 | 0.6555 | 0.5161 | 0.8977 | 0.0000 |
| QD-ER1_EduRisk_human_only | completed | 0.4379 | 0.5581 | 0.6700 | 0.4516 | 0.8036 | 0.0000 |
| QD-PR1_PairwiseRiskOrdinal_human_only | pending_formal_training | NA | NA | NA | NA | NA | NA |

QD-PR1 formal metrics are pending; this package records setup readiness only.
