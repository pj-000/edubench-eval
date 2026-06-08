# Exp5 L2 Notion Summary

问题：在不改变 A4 输入、不改变 ordinal objective、不使用 class weights 的前提下，
如果专门惩罚真实低分样本被预测得过高，能否进一步降低 low-to-high rate？

L2 和 L1 的区别：

- L1 处理类别不均衡，使用 class weights。
- L2 不使用 class weights，只加入低分高估非对称惩罚。
- L2a 使用 lambda_low=0.3，margin=0.0。
- L2b 使用 lambda_low=0.5，margin=0.0。

当前汇总：

| loss | status | Accuracy | MAE_label | MAE_expected | QWK | Kendall tau | low_to_high_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L0_exp04_o3_ordinal | completed | 0.7381 | 0.3777 | 0.3430 | 0.7036 | 0.6238 | 0.2330 |
| L1_weighted_ordinal | completed | 0.7250 | 0.3894 | 0.3504 | 0.7132 | 0.6149 | 0.2136 |
| L2a_asymmetric_ordinal_lambda03_margin0 | completed | 0.7160 | 0.4080 | 0.3705 | 0.6356 | 0.5991 | 0.3398 |
| L2b_asymmetric_ordinal_lambda05_margin0 | completed | 0.7164 | 0.3971 | 0.3633 | 0.6745 | 0.6038 | 0.2718 |

解读要点：

- 如果 L2 的 low_to_high_rate 低于 L1，说明方向性惩罚有额外作用。
- 如果 L2 同时损伤 Acc@5 或 high-score 指标，后续需要 L4 high-score preservation。
- L3/L4 暂不实现，等 L2 结果审阅后再做。
