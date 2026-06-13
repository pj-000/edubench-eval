# Exp9 QD-PR1 Pairwise Ordinal Review Package

Formal training status: `completed`
Training executed by setup stage: `no`.
API called: `no`.
Synthetic generated: `no`.

## Method

QD-PR1 adds risk-aware ordinal preference pairs on top of QD-B1-style weighted ordinal BCE.
Pairs are built from QD-S0 human-only train rows; dev pairs are diagnostic only.

## Formal Run Defaults

The formal run keeps `epochs=10`, `train_pairs=20000`, `dev_pairs=5000`, and
`effective_batch_size=128` unchanged. For 24GB RTX 3090 execution, the default micro-batch is
`per_device_train_batch_size=32` with `gradient_accumulation_steps=4`,
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

## Result Interpretation

QD-PR1 formal status is `completed`, but the formal result is negative and should not be presented
as an effective method.

- QD-PR1 does not reduce low_to_high relative to QD-B1.
- QD-PR1 does not beat QD-B1 on the main test metrics.
- Pairwise supervision remains promising, but this run damaged pointwise ordinal monotonicity and calibration.
- The next pairwise attempt should be anchored fine-tuning from QD-B1, not another unanchored from-scratch run.

| model | low_to_high | MAE_label | QWK | Acc@5 | monotonic_violation |
| --- | ---: | ---: | ---: | ---: | ---: |
| QD-B1 | 0.4516 | 0.4279 | 0.6012 | 0.7419 | 0.3119 |
| QD-PR1 | 0.5161 | 0.4545 | 0.5473 | 0.6916 | 0.7616 |

## Test Comparison

| run | status | MAE_label | QWK | Accuracy | low_to_high | Acc@5 | monotonic_violation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QD-B0_human_only_ordinary_ordinal | completed | 0.4019 | 0.5976 | 0.6936 | 0.5161 | 0.7873 | 0.1985 |
| QD-B1_human_only_L1_weighted_ordinal | completed | 0.4279 | 0.6012 | 0.6709 | 0.4516 | 0.7419 | 0.3119 |
| QD-R1_CORAL_human_only | completed | 0.4467 | 0.5272 | 0.6555 | 0.5161 | 0.8977 | 0.0000 |
| QD-ER1_EduRisk_human_only | completed | 0.4379 | 0.5581 | 0.6700 | 0.4516 | 0.8036 | 0.0000 |
| QD-PR1_PairwiseRiskOrdinal_human_only | completed | 0.4545 | 0.5473 | 0.6401 | 0.5161 | 0.6916 | 0.7616 |

## Deltas

| comparison | metric | delta | direction |
| --- | --- | ---: | --- |
| QD-PR1_minus_QD-B0 | low_to_high_rate | 0.0000 | lower_better |
| QD-PR1_minus_QD-B0 | MAE_label | 0.0526 | lower_better |
| QD-PR1_minus_QD-B0 | MAE_expected | 0.0547 | lower_better |
| QD-PR1_minus_QD-B0 | Accuracy | -0.0535 | higher_better |
| QD-PR1_minus_QD-B0 | Quadratic Weighted Kappa | -0.0503 | higher_better |
| QD-PR1_minus_QD-B0 | Kendall tau | -0.0748 | higher_better |
| QD-PR1_minus_QD-B0 | Spearman rho | -0.0796 | higher_better |
| QD-PR1_minus_QD-B0 | Acc@5 | -0.0958 | higher_better |
| QD-PR1_minus_QD-B0 | high_to_mid_or_low_rate | 0.0114 | lower_better |
| QD-PR1_minus_QD-B0 | monotonic_violation_rate | 0.5630 | lower_better |
| QD-PR1_minus_QD-B1 | low_to_high_rate | 0.0645 | lower_better |
| QD-PR1_minus_QD-B1 | MAE_label | 0.0266 | lower_better |
| QD-PR1_minus_QD-B1 | MAE_expected | 0.0330 | lower_better |
| QD-PR1_minus_QD-B1 | Accuracy | -0.0308 | higher_better |
| QD-PR1_minus_QD-B1 | Quadratic Weighted Kappa | -0.0539 | higher_better |
| QD-PR1_minus_QD-B1 | Kendall tau | -0.0721 | higher_better |
| QD-PR1_minus_QD-B1 | Spearman rho | -0.0814 | higher_better |
| QD-PR1_minus_QD-B1 | Acc@5 | -0.0503 | higher_better |
| QD-PR1_minus_QD-B1 | high_to_mid_or_low_rate | 0.0000 | lower_better |
| QD-PR1_minus_QD-B1 | monotonic_violation_rate | 0.4497 | lower_better |
| QD-PR1_minus_QD-R1 | low_to_high_rate | 0.0000 | lower_better |
| QD-PR1_minus_QD-R1 | MAE_label | 0.0079 | lower_better |
| QD-PR1_minus_QD-R1 | MAE_expected | 0.0355 | lower_better |
| QD-PR1_minus_QD-R1 | Accuracy | -0.0154 | higher_better |
| QD-PR1_minus_QD-R1 | Quadratic Weighted Kappa | 0.0201 | higher_better |
| QD-PR1_minus_QD-R1 | Kendall tau | -0.0011 | higher_better |
| QD-PR1_minus_QD-R1 | Spearman rho | -0.0017 | higher_better |
| QD-PR1_minus_QD-R1 | Acc@5 | -0.2062 | higher_better |
| QD-PR1_minus_QD-R1 | high_to_mid_or_low_rate | 0.0114 | lower_better |
| QD-PR1_minus_QD-R1 | monotonic_violation_rate | 0.7616 | lower_better |
| QD-PR1_minus_QD-ER1 | low_to_high_rate | 0.0645 | lower_better |
| QD-PR1_minus_QD-ER1 | MAE_label | 0.0166 | lower_better |
| QD-PR1_minus_QD-ER1 | MAE_expected | -0.0679 | lower_better |
| QD-PR1_minus_QD-ER1 | Accuracy | -0.0299 | higher_better |
| QD-PR1_minus_QD-ER1 | Quadratic Weighted Kappa | -0.0108 | higher_better |
| QD-PR1_minus_QD-ER1 | Kendall tau | -0.0149 | higher_better |
| QD-PR1_minus_QD-ER1 | Spearman rho | -0.0190 | higher_better |
| QD-PR1_minus_QD-ER1 | Acc@5 | -0.1120 | higher_better |
| QD-PR1_minus_QD-ER1 | high_to_mid_or_low_rate | -0.0097 | lower_better |
| QD-PR1_minus_QD-ER1 | monotonic_violation_rate | 0.7616 | lower_better |
