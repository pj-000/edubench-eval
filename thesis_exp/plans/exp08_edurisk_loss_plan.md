# Exp8 EduRisk Ordinal Loss Plan

Status: planning only. No training code, model training, API calls, synthetic data generation, or
changes to Exp0-Exp7 results are included in this plan.

## Goal

Design a training loss innovation for low-score risk-sensitive educational ordinal scoring:
`EduRisk Ordinal Loss`. The first target run is:

`QD-ER1_EduRisk_human_only`

The method should keep the CORAL-style rank-consistent cumulative structure from Exp7, but train it
with a richer objective that explicitly combines ordinal softness, low-score risk, long-tail class
balance, and cumulative-threshold stability.

## Motivation From Earlier Experiments

- Exp3/Exp4 showed that the A4 input setting is the strongest input representation.
- Exp6 showed that question-disjoint ordinal training is stronger than plain classification, but
  low-score overestimation remains.
- Exp5 L1 weighting gave only small improvements.
- Exp5 L2/L3 hand-crafted penalties and synthetic low-score augmentation were unstable.
- Exp7 CORAL-style rank consistency solved monotonic violations, but did not remove high-score bias.
- Exp7-C selective calibration reduced risk with human review, but it is not a training method.

EduRisk is therefore framed as a training-time loss innovation for the same question-disjoint,
human-only setting.

## Literature Grounding

EduRisk is motivated by the following lines of work:

- SLACE, AAAI 2025: balance-sensitive and monotone ordinal loss design motivates combining ordinal
  monotonicity with class imbalance awareness.
- ORCU, 2024/2025: ordinal calibration and unimodality via soft ordinal encodings motivates soft
  ordinal targets instead of one-hot class targets.
- Class-Balanced Loss: effective-number weighting motivates class weights that depend on class
  frequency without naive inverse-frequency explosion.
- Balanced Softmax, LDAM, and Focal Loss: long-tail classification references motivate separating
  imbalance correction from the task-specific ordinal risk term.
- CLOC, CVPR 2025: ordinal contrastive learning with multi-margin loss is a future extension, not
  part of the first EduRisk run.

Reference links:

- SLACE: https://ojs.aaai.org/index.php/AAAI/article/view/34158
- ORCU: https://openreview.net/forum?id=v27yHgKtMv
- Class-Balanced Loss: https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html
- Balanced Softmax: https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html
- LDAM: https://papers.nips.cc/paper/8435-learning-imbalanced-datasets-with-label-distribution-aware-margin-loss
- Focal Loss: https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html
- CLOC: https://openaccess.thecvf.com/content/CVPR2025/papers/Pitawela_CLOC_Contrastive_Learning_for_Ordinal_Classification_with_Multi-Margin_N-pair_Loss_CVPR_2025_paper.pdf

## Model Interface

Use a CORAL-style cumulative ordinal head with 4 cumulative probabilities for 5 labels:

```text
p_1 = P(y > 1)
p_2 = P(y > 2)
p_3 = P(y > 3)
p_4 = P(y > 4)
```

Rank consistency should enforce:

```text
1 >= p_1 >= p_2 >= p_3 >= p_4 >= 0
```

The cumulative probabilities are converted into a valid 5-class distribution:

```text
q_1 = 1 - p_1
q_2 = p_1 - p_2
q_3 = p_2 - p_3
q_4 = p_3 - p_4
q_5 = p_4
```

This makes the loss class-distribution aware while keeping rank-consistent cumulative modeling.

## EduRisk Loss Components

For a true label `y in {1,2,3,4,5}` and class index `k`, define a soft ordinal target:

```text
s_y(k) = exp(-abs(k - y) / tau) / Z_y
```

where:

```text
Z_y = sum_{j=1}^5 exp(-abs(j - y) / tau)
```

The soft ordinal cross entropy is:

```text
L_softCE = - sum_{k=1}^5 s_y(k) log(q_k)
```

The education-risk cost is normalized to keep the risk term on a scale comparable to CE/BCE:

```text
C_base(y,k) = abs(y-k) / 4
C_LH(y,k) = I(y <= 2 and k >= 4) * ((k-y) / 4)^2
C_HL(y,k) = I(y = 5 and k <= 3) * ((5-k) / 4)^2

C(y,k) = C_base(y,k)
       + lambda_LH * C_LH(y,k)
       + lambda_HL * C_HL(y,k)
```

The expected risk under the model distribution is:

```text
L_risk = sum_{k=1}^5 q_k C(y,k)
```

The cumulative BCE stability term reuses the CORAL threshold targets:

```text
t_m(y) = I(y > m), for m in {1,2,3,4}
L_cumBCE = (1/4) * sum_{m=1}^4 BCE(t_m(y), p_m)
```

Class-balanced weighting uses the effective number of samples:

```text
E_y = (1 - class_balance_beta ^ n_y) / (1 - class_balance_beta)
a_y = 1 / E_y
w_y = 5 * a_y / sum_{c=1}^5 a_c
```

The total loss is:

```text
L_EduRisk = w_y * [L_softCE + alpha * L_risk + beta_bce * L_cumBCE]
```

## First Default Hyperparameters

```text
tau = 0.7
alpha = 0.3
beta_bce = 0.5
lambda_LH = 2.0
lambda_HL = 0.5
class_balance_beta = 0.99
normalized_cost = true
decode_primary = cumulative
decode_secondary = argmax_q
```

Interpretation:

- `tau=0.7` gives nearby ordinal classes nonzero target mass without becoming too flat.
- `alpha=0.3` makes risk visible but avoids dominating the ordinal likelihood.
- `beta_bce=0.5` preserves cumulative threshold stability.
- `lambda_LH=2.0` prioritizes low-score overestimation errors.
- `lambda_HL=0.5` protects against severe underestimation of true high-score answers, but less
  aggressively than low-to-high risk.
- `class_balance_beta=0.99` follows effective-number weighting but avoids nearly suppressing label
  4/5 samples in the imbalanced question_seed42 human-only train distribution.
- `normalized_cost=true` avoids the unnormalized `y=1,k=5` risk cost becoming much larger than CE/BCE.

Approximate first-run class-balanced weights under question_seed42 human-only train:

| label | train count | beta=0.99 approximate weight |
| --- | ---: | ---: |
| 1 | 58 | 1.46 |
| 2 | 53 | 1.56 |
| 3 | 297 | 0.68 |
| 4 | 1163 | 0.65 |
| 5 | 1755 | 0.65 |

Do not use `class_balance_beta=0.999` in the first run. If a later run uses it, the config must add
`weight_clip_min=0.5` and `weight_clip_max=3.0`.

## Required Config Fields

The first implementation must put every EduRisk parameter in the run config, not hard-code it:

```yaml
tau: 0.7
alpha_risk: 0.3
beta_bce: 0.5
lambda_lh: 2.0
lambda_hl: 0.5
class_balance_beta: 0.99
normalized_cost: true
decode_primary: cumulative
decode_secondary: argmax_q
```

## First Run Setup

- Dataset: question-disjoint `question_seed42`, human-only.
- Input: A4 representation.
- Synthetic data: no.
- Run id: `QD-ER1_EduRisk_human_only`.
- Baselines:
  - `QD-B0_human_only_ordinary_ordinal`
  - `QD-B1_human_only_L1_weighted_ordinal`
  - `QD-R1_CORAL_human_only`
  - `QD-ER1_EduRisk_human_only`
- Selection discipline:
  - Hyperparameters are fixed before the first run.
  - Dev is used only for model selection/early stopping in the standard pipeline.
  - Test is used once for final evaluation.

## Metrics

Primary risk metric:

```text
low_to_high_rate = P(pred_label >= 4 | true_label <= 2)
```

Secondary metrics:

- MAE
- QWK
- Kendall tau
- Acc@1
- Acc@2
- Acc@5
- signed bias, especially low-score signed bias
- monotonic_violation_rate
- expected risk using the EduRisk cost matrix

Decoding diagnostics must be saved for every split:

```text
pred_label_cum = 1 + sum_m I(p_m > 0.5)
pred_label_argmax = argmax_{k in 1..5} q_k
pred_score_expected = sum_k k * q_k
```

The main comparison table uses `pred_label_cum` for fair comparison with QD-B0, QD-B1, and QD-R1.
An additional diagnostic table must compare cumulative versus argmax decoding:

```text
cumulative_decoding_vs_argmax_decoding.csv
```

## Success Criteria

EduRisk is considered a useful training method if it satisfies all of the following on test:

- `monotonic_violation_rate = 0`, matching CORAL rank consistency.
- `low_to_high_rate < QD-B1 low_to_high_rate`.
- MAE is no worse than `QD-B1 + 0.02`.
- QWK is no worse than `QD-B1 - 0.03`.
- Acc@5 is no worse than `QD-B1 - 0.05`.
- `high_to_mid_or_low_rate` is no worse than `QD-B1 + 0.03`.
- Low-score signed bias is closer to zero than QD-R1.

Stretch target:

- `low_to_high_rate <= 0.40` without human review.

## Why This Is Different From Exp5 L1/L2/L3

Exp5 used hand-crafted scalar reweighting or penalties layered on top of the prior objective. EduRisk
changes the training target and the expected loss geometry:

- It uses a legal class distribution `q` derived from CORAL cumulative probabilities.
- It trains with soft ordinal targets, not one-hot or threshold-only supervision.
- It computes expected task risk over all classes, not just a penalty attached to the observed
  prediction.
- It uses effective-number class balancing rather than ad hoc low-label weights.
- It retains cumulative BCE as a stability term instead of relying on penalties alone.
- Its risk cost is normalized before weighting, so `alpha` has a stable interpretation.

## Risks And Guardrails

- The risk term may over-suppress high predictions. Guard against QD-B1 Acc@5 and
  high-to-mid-or-low metrics, not QD-R1's high-biased Acc@5.
- Effective-number weights may over-amplify rare labels. Use `class_balance_beta=0.99` for the first
  run and check weight range before training.
- Soft targets may blur adjacent classes. Guard with MAE and QWK.
- The cumulative BCE term may dominate if scaled too high. Keep `beta_bce=0.5` for the first run.
- Do not combine with synthetic augmentation in the first run.

## Required Loss-scale Logging

Code stage must log all loss components on train and dev:

```text
L_total
L_softCE
L_risk
L_cumBCE
mean_weight
min_weight
max_weight
weighted_L_softCE
weighted_L_risk
weighted_L_cumBCE
```

If `weighted_L_risk` is much larger than `weighted_L_softCE`, the run should be flagged for review
before being interpreted as a scientific result.

## Next Step

Only after this plan is reviewed, code stage can implement:

- loss module,
- config entry,
- smoke test,
- full run script,
- output sanity checks,
- no-training dry-run checks before any real training launch.
