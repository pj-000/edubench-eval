# Exp8 EduRisk Ordinal Loss Summary

Status: planning only.

## Method

EduRisk Ordinal Loss is a proposed full-coverage training method for low-score risk-sensitive
educational ordinal scoring.

It keeps a CORAL-style cumulative ordinal head, converts cumulative probabilities into a valid
5-class distribution, and trains with:

- soft ordinal target cross entropy,
- expected low-score risk cost,
- effective-number class balancing,
- optional cumulative BCE stability.

## Formula

```text
q_1 = 1 - p_1
q_2 = p_1 - p_2
q_3 = p_2 - p_3
q_4 = p_3 - p_4
q_5 = p_4

s_y(k) = exp(-abs(k-y)/tau) / Z_y

C_base(y,k) = abs(y-k) / 4
C_LH(y,k) = I(y <= 2 and k >= 4) * ((k-y) / 4)^2
C_HL(y,k) = I(y = 5 and k <= 3) * ((5-k) / 4)^2
C(y,k) = C_base + lambda_LH * C_LH + lambda_HL * C_HL

L_EduRisk = w_y * [L_softCE + alpha * L_risk + beta_bce * L_cumBCE]
```

## First Run

```text
run_id = QD-ER1_EduRisk_human_only
data = question_seed42 human-only
input = A4
synthetic = no
```

Default hyperparameters:

```text
tau = 0.7
alpha_risk = 0.3
beta_bce = 0.5
lambda_LH = 2.0
lambda_HL = 0.5
class_balance_beta = 0.99
normalized_cost = true
decode_primary = cumulative
decode_secondary = argmax_q
```

## Baselines

- QD-B0 ordinary ordinal
- QD-B1 L1-weighted ordinal
- QD-R1 CORAL
- QD-ER1 EduRisk

## Success Criteria

EduRisk succeeds if it lowers low_to_high versus QD-B1 while preserving MAE, QWK, Acc@5, and
high-score guardrails relative to QD-B1.

Stretch target:

```text
low_to_high_rate <= 0.40
```

## Positioning

Exp7-C selective calibration remains the human-in-the-loop fallback. EduRisk is the proposed
training-method innovation for a full-coverage ordinal scorer.

## Next Step

Review the plan first. Code stage can start after approval.

Code stage must also log loss component scales and save both `pred_label_cum` and
`pred_label_argmax` diagnostics.
