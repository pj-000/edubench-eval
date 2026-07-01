# Exp17-A1 Evidence Head Scout Report

Exp17-A1 keeps the Exp16A qmr ordinal decision function unchanged while adding a hidden-failure evidence head `h`. Joint A1 configs fine-tune the base model with an auxiliary evidence objective; A1F is a frozen-base probe that trains only the evidence head.

## Guardrails

- Test split is not read.
- Dev D1 annotations are used only for dev evidence evaluation.
- Human rationale text is not used as model input.
- The evidence head does not suppress or alter `s`.
- `A1_0_baseline` is a continued-training ordinal control, not the frozen original Exp16A checkpoint.
- `A1_6_random_positive_control` samples random low-label positives, not arbitrary non-low positives.

## Completed Configs

| config | beta | neg_ratio | MAE | QWK | low-to-high | label2 recall | h AUC | hidden-control delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `A1F_2_frozen_probe_lr1em3_gradaccum1_epochs20` | 0.1000 | 1:4 | 0.3993 | 0.5532 | 0.5088 | 0.0000 | 0.2874 | -0.0040 |
| `A1F_3_frozen_probe_lr3em4_gradaccum1_epochs20` | 0.1000 | 1:4 | 0.3993 | 0.5532 | 0.5088 | 0.0000 | 0.3798 | -0.0220 |
| `A1F_4_frozen_probe_lr1em4_gradaccum1_epochs30` | 0.1000 | 1:4 | 0.3993 | 0.5532 | 0.5088 | 0.0000 | 0.4402 | -0.0169 |
| `A1_1b_a0_weak_random_high_negatives` | 0.1000 | 1:4 | 0.4020 | 0.5437 | 0.5439 | 0.0000 | 0.5593 | 0.0083 |
| `A1_5a_all_low_downsample76_same_neg_pool` | 0.1000 | 1:4 | 0.3957 | 0.5664 | 0.5088 | 0.0000 | 0.4092 | -0.0458 |
| `A1_5b_all_low111_same_clean_high_controls` | 0.1000 | 1:4 | 0.3921 | 0.5709 | 0.5088 | 0.0000 | 0.5123 | 0.0027 |

## Decision

- best_config: `A1_1b_a0_weak_random_high_negatives`
- A1 success: `False`
- MAE degradation vs A1_0_baseline: NA
- QWK degradation vs A1_0_baseline: NA
- hidden-vs-control AUC: 0.5593
- hidden-control h delta: 0.0083
- beats all-low and random controls: `True`

Proceed to C1 only if A1 succeeds. Keep B1 suppression delayed until the evidence head is reliable.