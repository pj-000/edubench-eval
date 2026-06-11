# Exp8 EduRisk Ordinal Loss Formulas

Status: formulas only. This document does not define or run training code.

## Notation

Let:

```text
K = 5
y in {1,2,3,4,5}
m in {1,2,3,4}
k in {1,2,3,4,5}
```

The model has a CORAL-style cumulative ordinal head:

```text
p_m = P(y > m)
```

The first Exp8 run assumes rank-consistent cumulative probabilities:

```text
1 >= p_1 >= p_2 >= p_3 >= p_4 >= 0
```

## Cumulative-to-class Conversion

Convert cumulative probabilities into a 5-class distribution:

```text
q_1 = 1 - p_1
q_2 = p_1 - p_2
q_3 = p_2 - p_3
q_4 = p_3 - p_4
q_5 = p_4
```

If rank consistency holds, then:

```text
q_k >= 0 for all k
sum_{k=1}^5 q_k = 1
```

Implementation guard for future code:

```text
q = clamp(q, eps, 1)
q = q / sum(q)
```

This guard should be used only for numerical stability; a nonzero monotonic violation rate should
still fail the sanity check.

## Soft Ordinal Target

Define:

```text
u_y(k) = exp(-abs(k-y) / tau)
Z_y = sum_{j=1}^5 u_y(j)
s_y(k) = u_y(k) / Z_y
```

Properties:

- `s_y(y)` is the largest target mass.
- Adjacent labels receive nonzero mass.
- Far labels are discouraged but not treated as equally wrong.

Soft ordinal cross entropy:

```text
L_softCE(y,q) = - sum_{k=1}^5 s_y(k) log(q_k)
```

Default:

```text
tau = 0.7
```

## Education-risk Cost

Base ordinal error:

```text
C_base(y,k) = abs(y-k)
```

Low-score overestimation penalty:

```text
C_LH(y,k) = I(y <= 2 and k >= 4) * (k-y)^2
```

High-score underestimation penalty:

```text
C_HL(y,k) = I(y = 5 and k <= 3) * (5-k)^2
```

Total cost:

```text
C(y,k) = C_base(y,k)
       + lambda_LH * C_LH(y,k)
       + lambda_HL * C_HL(y,k)
```

Defaults:

```text
lambda_LH = 2.0
lambda_HL = 0.5
```

Risk loss:

```text
L_risk(y,q) = sum_{k=1}^5 q_k C(y,k)
```

This term penalizes probability mass assigned to high-risk wrong labels, even before argmax or
threshold decoding.

## Class-balanced Weight

For class count `n_y`, define the effective number of samples:

```text
E_y = (1 - cb_beta ^ n_y) / (1 - cb_beta)
```

Raw inverse effective-number weight:

```text
a_y = 1 / E_y
```

Normalize across labels:

```text
w_y = 5 * a_y / sum_{c=1}^5 a_c
```

Default:

```text
cb_beta = 0.999
```

Guardrails:

- Report `min(w_y)`, `max(w_y)`, and `max(w_y) / min(w_y)` before training.
- If the ratio is too large, cap weights or lower `cb_beta` in a later ablation, not in the first
  default run.

## Cumulative BCE Stability Term

The CORAL target for threshold `m` is:

```text
t_m(y) = I(y > m)
```

The cumulative BCE term is:

```text
L_cumBCE(y,p) = (1/4) * sum_{m=1}^4 [
  - t_m(y) log(p_m)
  - (1 - t_m(y)) log(1 - p_m)
]
```

Default:

```text
beta_bce = 0.5
```

Purpose:

- Keeps the cumulative thresholds anchored.
- Reduces the risk that soft targets and expected risk alone destabilize the ordinal head.

## Total EduRisk Loss

The final per-sample loss is:

```text
L_EduRisk =
  w_y * [
    L_softCE(y,q)
    + alpha * L_risk(y,q)
    + beta_bce * L_cumBCE(y,p)
  ]
```

First default:

```text
tau = 0.7
alpha = 0.3
beta_bce = 0.5
lambda_LH = 2.0
lambda_HL = 0.5
cb_beta = 0.999
```

## Expected Risk Evaluation Metric

For evaluation, report the model-expected education risk:

```text
ExpectedEduRisk = mean_i sum_{k=1}^5 q_{i,k} C(y_i,k)
```

Also report risk by true label group:

```text
ExpectedEduRisk_low = mean over y <= 2
ExpectedEduRisk_mid = mean over y in {3,4}
ExpectedEduRisk_high = mean over y = 5
```

These are diagnostic metrics and should not replace MAE, QWK, or low_to_high.

## Decoding

For consistency with prior ordinal runs, primary decoded prediction should use the cumulative
threshold rule:

```text
pred_label = 1 + sum_{m=1}^4 I(p_m > 0.5)
```

For analysis, also allow expected-score diagnostics:

```text
pred_score_expected = sum_{k=1}^5 k * q_k
```

The headline metrics should use `pred_label` so comparisons to QD-B0, QD-B1, and QD-R1 remain fair.
