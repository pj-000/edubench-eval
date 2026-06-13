# QD-PR2 Formal Summary

Formal status: completed.
Training in this diagnosis step: no.
API called: no.
Synthetic generated: no.

## Main Result

| 指标 | 低分样本加权序数评分基线 | 锚定式成对边界微调 | 变化 |
| --- | ---: | ---: | ---: |
| low-to-high | 0.4516 | 0.3871 | -0.0645 |
| MAE | 0.4279 | 0.4192 | -0.0088 |
| QWK | 0.6012 | 0.6084 | +0.0071 |
| Accuracy | 0.6709 | 0.6772 | +0.0063 |
| Acc@5 | 0.7419 | 0.7549 | +0.0130 |
| monotonic violation | 0.3119 | 0.3527 | +0.0408 |

## Interpretation

QD-PR2 reduces low-to-high errors by `2` cases on the test split (`14` -> `12` among `31` low-score cases).
It improves MAE, QWK, Accuracy, and Acc@5, so it is the strongest current training-method result. It still worsens monotonic violation
(`0.3119` -> `0.3527`), mainly through `p_gt_1 < p_gt_2`.

## Recommendation

Use QD-PR2 in the midterm as a promising training innovation, but state clearly that it is not a fully solved final scorer because threshold monotonicity remains defective.
