# Exp9 QD-PR1 Pairwise Ordinal Review Package

Formal training status: `pending_formal_training`
Training executed by setup stage: `no`.
API called: `no`.
Synthetic generated: `no`.

## Method

QD-PR1 adds risk-aware ordinal preference pairs on top of QD-B1-style weighted ordinal BCE.
Pairs are built from QD-S0 human-only train rows; dev pairs are diagnostic only.

## Formal Run Defaults

The formal run keeps `epochs=10`, `train_pairs=20000`, `dev_pairs=5000`, and
`effective_batch_size=128` unchanged. For 24GB RTX 3090 execution, the default micro-batch is
`per_device_train_batch_size=16` with `gradient_accumulation_steps=8`,
`per_device_eval_batch_size=8`, `max_length=2048`, and `gradient_checkpointing=true`. The train
script ignores inherited hyperparameter environment variables by default; set
`ALLOW_EXP09_ENV_OVERRIDES=1` only for deliberate manual overrides.

## Pair Comparability Audit

Pair construction follows priority `same_question > same_metric_language > same_metric`; an
`any_valid` fallback is used only if needed to fill the configured pair budget.
Actual comparability rates are reported below. Formal training results should be interpreted in
light of these rates.

| split | pair_type | pairs | same_question | same_metric | same_language | same_metric_language |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | low_high | 8000 | 0.2604 | 0.7550 | 1.0000 | 0.7550 |
| train | low_mid | 4000 | 0.0895 | 0.8260 | 0.5225 | 0.3970 |
| train | adjacent | 6000 | 1.0000 | 0.1208 | 1.0000 | 0.1208 |
| train | random_ordinal | 2000 | 1.0000 | 0.0795 | 1.0000 | 0.0795 |
| dev | low_high | 2000 | 0.2850 | 0.7355 | 1.0000 | 0.7355 |
| dev | low_mid | 1000 | 0.2780 | 0.6050 | 0.6770 | 0.3580 |
| dev | adjacent | 1500 | 1.0000 | 0.1067 | 1.0000 | 0.1067 |
| dev | random_ordinal | 500 | 1.0000 | 0.0800 | 1.0000 | 0.0800 |

## Test Comparison

| run | status | MAE_label | QWK | Accuracy | low_to_high | Acc@5 | monotonic_violation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QD-B0_human_only_ordinary_ordinal | completed | 0.4019 | 0.5976 | 0.6936 | 0.5161 | 0.7873 | 0.1985 |
| QD-B1_human_only_L1_weighted_ordinal | completed | 0.4279 | 0.6012 | 0.6709 | 0.4516 | 0.7419 | 0.3119 |
| QD-R1_CORAL_human_only | completed | 0.4467 | 0.5272 | 0.6555 | 0.5161 | 0.8977 | 0.0000 |
| QD-ER1_EduRisk_human_only | completed | 0.4379 | 0.5581 | 0.6700 | 0.4516 | 0.8036 | 0.0000 |
| QD-PR1_PairwiseRiskOrdinal_human_only | pending_formal_training | NA | NA | NA | NA | NA | NA |

QD-PR1 formal metrics are pending; this package records setup readiness only.
