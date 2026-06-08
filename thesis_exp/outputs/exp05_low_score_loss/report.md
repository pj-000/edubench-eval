# Exp5 Low-Score Loss Ablation

Exp5 studies whether the low-score overestimation problem is partly caused by
class imbalance in the train split.

Exp4 already selected A4 plus ordinal classification as the strongest overall
scoring setup. However, the Exp4 O3 ordinal baseline still overestimates some
low-score samples. L1 is a control experiment for class imbalance, not a new
main method.

L1 keeps the input, objective, data split, model backbone, and checkpoint
selection comparable to Exp4 O3. The only change is the sample weight:

`weight_c = clip(N / (5 * N_c), 0.5, 3.0)`

The weighted loss is:

`sum_i(w_i * L_i_ord) / sum_i(w_i)`

L1 does not explicitly punish the direction where low scores are predicted as
high scores. That asymmetric error will be studied later in L2/L3.

## Class Weights

| label_5 | train_count | raw_weight | clipped_weight |
| ---: | ---: | ---: | ---: |
| 1 | 24 | 22.1167 | 3.0000 |
| 2 | 52 | 10.2077 | 3.0000 |
| 3 | 251 | 2.1147 | 2.1147 |
| 4 | 946 | 0.5611 | 0.5611 |
| 5 | 1381 | 0.3844 | 0.5000 |

## Main Results

| loss | status | Accuracy | MAE_label | MAE_expected | QWK | Kendall tau | low_to_high_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L0_exp04_o3_ordinal | completed | 0.7381 | 0.3777 | 0.3430 | 0.7036 | 0.6238 | 0.2330 |
| L1_weighted_ordinal | pending | NA | NA | NA | NA | NA | NA |

## Output Files

- Summary table: `thesis_exp/outputs/exp05_low_score_loss/tables/loss_ablation_summary.csv`
- Low-score table: `thesis_exp/outputs/exp05_low_score_loss/tables/loss_ablation_low_score.csv`
- Per-bin table: `thesis_exp/outputs/exp05_low_score_loss/tables/loss_ablation_per_bin.csv`
- Delta table: `thesis_exp/outputs/exp05_low_score_loss/tables/loss_ablation_delta_vs_L0.csv`
