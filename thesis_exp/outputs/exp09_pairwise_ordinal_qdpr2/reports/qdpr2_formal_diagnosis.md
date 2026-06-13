# QD-PR2 Formal Result Diagnosis

Run: `QD-PR2_AnchoredPairwiseOrdinal_human_only`.
Formal status: `completed`.
Training run in this diagnosis step: `no`.
API called: `no`.
Synthetic generated: `no`.
Raw predictions/arrays/logs modified: `no`.

## Executive Finding

QD-PR2 should be described as a promising training-method improvement, not as a fully solved final scorer.
Compared with QD-B1, it improves low-to-high, MAE, QWK, Accuracy, and Acc@5 on the test split, but the
monotonic violation rate is worse.

| 指标 | 低分样本加权序数评分基线 | 锚定式成对边界微调 | 变化 |
| --- | ---: | ---: | ---: |
| low-to-high | 0.4516 | 0.3871 | -0.0645 |
| MAE | 0.4279 | 0.4192 | -0.0088 |
| QWK | 0.6012 | 0.6084 | +0.0071 |
| Accuracy | 0.6709 | 0.6772 | +0.0063 |
| Acc@5 | 0.7419 | 0.7549 | +0.0130 |
| monotonic violation | 0.3119 | 0.3527 | +0.0408 |

## Low-to-high Error Reduction

The test split has `31` true low-score samples (`label <= 2`). QD-B1 produces `14`
low-to-high errors, while QD-PR2 produces `12`. The absolute reduction is `2`
errors, from `0.4516` to `0.3871`.

- Resolved low-to-high cases: `2`.
- New low-to-high cases introduced by QD-PR2: `0`.
- Net reduction: `2`.

## Which True Labels Benefited Most?

| true label | n | baseline low-to-high | anchored low-to-high | delta count | baseline MAE | anchored MAE | MAE delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9 | 1 (0.1111) | 1 (0.1111) | +0 | 0.4444 | 0.4444 | 0.0000 |
| 2 | 22 | 13 (0.5909) | 11 (0.5000) | -2 | 1.7727 | 1.6818 | -0.0909 |
| 3 | 105 | 0 (0.0000) | 0 (0.0000) | +0 | 0.7619 | 0.7714 | 0.0095 |
| 4 | 351 | 0 (0.0000) | 0 (0.0000) | +0 | 0.3305 | 0.3305 | 0.0000 |
| 5 | 616 | 0 (0.0000) | 0 (0.0000) | +0 | 0.2760 | 0.2630 | -0.0130 |

The main low-score gain comes from label 2: low-to-high falls from `13`
to `11` cases. Label 1 is unchanged on
low-to-high count and MAE. Label 5 also improves in MAE/Acc@5, which suggests the anchor loss helps avoid the
high-score collapse seen in less constrained risk-aware variants.

## Does Label 2 Remain Difficult?

Yes. Label 2 remains difficult. Even after QD-PR2, label 2 still has `11`
low-to-high cases out of `22` label-2 examples on test. Its mean predicted label
remains `3.6818`, so the model still tends to overestimate label-2 answers.

## Monotonicity Defect

| threshold pair | baseline rate | anchored rate | anchored count | main affected true label counts |
| --- | ---: | ---: | ---: | --- |
| p_gt_1<p_gt_2 | 0.2747 | 0.3037 | 335 | y=2: 2, y=3: 12, y=4: 63, y=5: 258 |
| p_gt_2<p_gt_3 | 0.0399 | 0.0462 | 51 | y=1: 7, y=3: 6, y=4: 11, y=5: 27 |
| p_gt_3<p_gt_4 | 0.0091 | 0.0118 | 13 | y=1: 1, y=3: 3, y=4: 2, y=5: 7 |

The dominant violation is `p_gt_1 < p_gt_2`. This indicates a threshold-ordering defect in the independent ordinal head: the model often assigns a
slightly higher probability to passing the second threshold than to passing the first. The magnitudes are small, but the frequency is high enough to
raise the overall monotonic violation rate from `0.3119` to `0.3527`.

Why it remains worse: QD-PR2 anchors to QD-B1 and adds a monotonic penalty, but the pairwise objective operates on the scalar expected score
`r(x)=1+sum_t sigmoid(z_t)`. It can improve ordering margins without guaranteeing every adjacent cumulative threshold remains ordered.
The monotonic regularizer is too weak relative to the independent threshold degrees of freedom.

## Pairwise Gap Diagnostics

For dev low-high pairs, the mean score gap improves from `1.0548` in from-scratch pairwise training to
`1.3527` in anchored pairwise boundary fine-tuning. Low-high pairwise accuracy is essentially stable
(`0.7920` -> `0.7933`), while low-high margin satisfaction improves
(`0.4040` -> `0.4883`).

Overall pairwise accuracy also improves slightly (`0.7964` -> `0.8040`). This supports the view
that pairwise boundary learning works better when it is anchored to a stable pointwise ordinal scorer.

## Anchor Loss and Pointwise Stability

Anchor loss appears to stabilize pointwise metrics. QD-PR2 improves MAE (`0.4279` -> `0.4192`),
QWK (`0.6012` -> `0.6084`), and Acc@5
(`0.7419` -> `0.7549`), unlike QD-PR1 which damaged pointwise calibration.
The mean anchor contribution is `0.0642` and the mean pair contribution is
`0.0181`, both comparable to but not overwhelming the pointwise term.

## Should QD-PR2 Be the Current Main Training Innovation Result?

Yes, as the current main training-method innovation result, with careful wording. It is stronger than QD-B1 on the primary low-to-high metric and
several overall metrics, and it demonstrates that pairwise boundary learning needs a pointwise ordinal anchor. It should not be presented as a final
fully solved scorer because monotonicity is worse and still needs stronger threshold-consistency control.

## PPT Wording

Use these statements:

- “锚定式成对边界微调在低分高估和整体评分一致性上均优于低分样本加权序数评分基线。”
- “该方法仍存在序数单调性缺陷，后续需进一步加强阈值一致性约束。”
- “该结果表明，成对边界学习需要被稳定的点式评分模型锚定，不能从头单独训练。”
