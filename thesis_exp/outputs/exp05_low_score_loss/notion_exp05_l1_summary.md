# Exp5 L1 Notion Summary

Question: are low-score failures mainly caused by label imbalance?

L1 reuses Exp4 O3 ordinal as L0 and trains only a weighted ordinal variant.

| loss | status | Accuracy | MAE_label | MAE_expected | QWK | Kendall tau | low_to_high_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L0_exp04_o3_ordinal | completed | 0.7381 | 0.3777 | 0.3430 | 0.7036 | 0.6238 | 0.2330 |
| L1_weighted_ordinal | completed | 0.7250 | 0.3894 | 0.3504 | 0.7132 | 0.6149 | 0.2136 |
| L2a_asymmetric_ordinal_lambda03_margin0 | completed | 0.7160 | 0.4080 | 0.3705 | 0.6356 | 0.5991 | 0.3398 |
| L2b_asymmetric_ordinal_lambda05_margin0 | completed | 0.7164 | 0.3971 | 0.3633 | 0.6745 | 0.6038 | 0.2718 |

Class weights come from the train split only.

| label_5 | train_count | raw_weight | clipped_weight |
| ---: | ---: | ---: | ---: |
| 1 | 24 | 22.1167 | 3.0000 |
| 2 | 52 | 10.2077 | 3.0000 |
| 3 | 251 | 2.1147 | 2.1147 |
| 4 | 946 | 0.5611 | 0.5611 |
| 5 | 1381 | 0.3844 | 0.5000 |
