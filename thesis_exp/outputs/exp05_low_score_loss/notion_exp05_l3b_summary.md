# Exp5 L3b Notion Summary

问题：L2 的 expected-score penalty 没有降低 low_to_high_rate 后，是否应该改用和
low_to_high 更直接对齐的 threshold-level penalty？

L3b 做的事：

- 复用 L1 的 train-split class weights。
- base loss 是 normalized weighted ordinal BCE。
- 只对真实低分样本 label_5<=2 惩罚 `p_gt_3` 和 `p_gt_4`。
- 使用 `mu_thr=0.3`。
- 不使用 L2 expected-score penalty。
- 不实现 L3a、L4、synthetic data、calibration。

当前汇总：

| loss | status | Accuracy | MAE_label | MAE_expected | QWK | Kendall tau | low_to_high_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L0_exp04_o3_ordinal | completed | 0.7381 | 0.3777 | 0.3430 | 0.7036 | 0.6238 | 0.2330 |
| L1_weighted_ordinal | completed | 0.7250 | 0.3894 | 0.3504 | 0.7132 | 0.6149 | 0.2136 |
| L2a_asymmetric_ordinal_lambda03_margin0 | completed | 0.7160 | 0.4080 | 0.3705 | 0.6356 | 0.5991 | 0.3398 |
| L2b_asymmetric_ordinal_lambda05_margin0 | completed | 0.7164 | 0.3971 | 0.3633 | 0.6745 | 0.6038 | 0.2718 |
| L3b_weighted_threshold_mu03 | completed | 0.7205 | 0.3946 | 0.3602 | 0.6871 | 0.6122 | 0.2330 |

训练前检查：

- L3b toy loss checks: PASS
- setup sanity 和 readability 通过后，可先运行 L3b smoke test。
