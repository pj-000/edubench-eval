# Exp8 EduRisk Ordinal Loss Report

Status: `implementation_ready_training_not_run`

Training executed by this package: `no` until the formal train script is run.
API calls: `no`.
Synthetic data: `no`.

## Method

QD-ER1 uses a CORAL-style rank-consistent head and converts cumulative probabilities into a legal
5-class distribution.
The training objective combines soft ordinal cross entropy, normalized low-score risk,
effective-number class balancing, and cumulative BCE.

## Current Comparison

| run | status | low_to_high | MAE | QWK | Acc@5 | signed bias | monotonic violation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QD-B0_human_only_ordinary_ordinal | completed | 0.5161 | 0.4019 | 0.5976 | 0.7873 | 0.1070 | 0.1985 |
| QD-B1_human_only_L1_weighted_ordinal | completed | 0.4516 | 0.4279 | 0.6012 | 0.7419 | 0.0181 | 0.3119 |
| QD-R1_CORAL_human_only | completed | 0.5161 | 0.4467 | 0.5272 | 0.8977 | 0.2829 | 0.0000 |
| QD-ER1_EduRisk_human_only | implementation_ready_training_not_run | NA | NA | NA | NA | NA | NA |

## Guardrails

- Primary low-score target: QD-ER1 `low_to_high_rate` should be lower than QD-B1 (`0.4516`).
- MAE should be no worse than QD-B1 + 0.02.
- QWK should be no worse than QD-B1 - 0.03.
- Acc@5 should be no worse than QD-B1 - 0.05, not QD-R1.
- high_to_mid_or_low should be no worse than QD-B1 + 0.03.
- Signed bias should be closer to zero than QD-R1.
- monotonic_violation_rate must remain 0.

## Recommendation

Run server smoke first, then formal QD-ER1 only if the smoke sanity checks pass.
