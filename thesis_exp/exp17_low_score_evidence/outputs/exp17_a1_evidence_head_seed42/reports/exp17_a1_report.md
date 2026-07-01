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
| `A1F_1_frozen_base_beta_0p10` | 0.1000 | 1:4 | 0.3993 | 0.5532 | 0.5088 | 0.0000 | 0.4733 | 0.0127 |
| `A1_0_baseline` | 0.0000 | 1:4 | 0.3966 | 0.5507 | 0.4912 | 0.0000 | 0.4615 | 0.0054 |
| `A1_1` | 0.0500 | 1:4 | 0.3930 | 0.5310 | 0.4912 | 0.0000 | 0.3985 | -0.0118 |
| `A1_2` | 0.1000 | 1:4 | 0.3966 | 0.5591 | 0.5263 | 0.0000 | 0.4738 | -0.0061 |
| `A1_3` | 0.2000 | 1:4 | 0.3939 | 0.5743 | 0.5263 | 0.0000 | 0.5529 | 0.0152 |
| `A1_4` | 0.1000 | 1:2 | 0.3930 | 0.5530 | 0.4912 | 0.0000 | 0.3894 | -0.0176 |
| `A1_5_all_low_aux_baseline` | 0.1000 | 1:4 | 0.3866 | 0.5810 | 0.4912 | 0.0000 | 0.5732 | 0.0165 |
| `A1_6_random_positive_control` | 0.1000 | 1:4 | 0.4011 | 0.5694 | 0.4912 | 0.0000 | 0.3734 | -0.0278 |

## Decision

- best_config: `A1_3`
- A1 success: `False`
- MAE degradation vs A1_0_baseline: -0.0027
- QWK degradation vs A1_0_baseline: -0.0236
- hidden-vs-control AUC: 0.5529
- hidden-control h delta: 0.0152
- beats all-low and random controls: `False`

Proceed to C1 only if A1 succeeds. Keep B1 suppression delayed until the evidence head is reliable.