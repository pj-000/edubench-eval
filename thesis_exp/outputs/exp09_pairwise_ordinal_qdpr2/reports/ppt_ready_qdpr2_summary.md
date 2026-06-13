# PPT-ready Anchored Pairwise Boundary Fine-tuning Summary

## Result Table

| 指标 | 低分样本加权序数评分基线 | 锚定式成对边界微调 | 变化 |
| --- | ---: | ---: | ---: |
| low-to-high | 0.4516 | 0.3871 | -0.0645 |
| MAE | 0.4279 | 0.4192 | -0.0088 |
| QWK | 0.6012 | 0.6084 | +0.0071 |
| Accuracy | 0.6709 | 0.6772 | +0.0063 |
| Acc@5 | 0.7419 | 0.7549 | +0.0130 |
| monotonic violation | 0.3119 | 0.3527 | +0.0408 |

## One-slide Takeaway

锚定式成对边界微调在低分高估和整体评分一致性上均优于低分样本加权序数评分基线。

该方法仍存在序数单调性缺陷，后续需进一步加强阈值一致性约束。

该结果表明，成对边界学习需要被稳定的点式评分模型锚定，不能从头单独训练。

## Speaker Notes

- 低分高估错误从 `14` 个降到 `12` 个，净减少 `2` 个。
- MAE、QWK、Accuracy 和 Acc@5 同时改善，说明这不是只牺牲整体指标换来的局部改进。
- 但 monotonic violation 从 `0.3119` 升到 `0.3527`，所以不要说“已经解决序数一致性”。
- 推荐表述为“promising training-method improvement with a remaining monotonicity defect”。
