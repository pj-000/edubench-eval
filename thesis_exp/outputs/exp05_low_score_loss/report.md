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
high scores.

## L2 Asymmetric Low-Score Overestimation Loss

L2 tests a different hypothesis: if true low-score samples are predicted too
high, explicitly penalize that overestimation during ordinal training. L2 does
not use class weights, does not change the A4 input, does not change the data
split, and does not add high-score preservation or threshold suppression.

For ordinal probabilities `p_t = sigmoid(z_t)`, the continuous prediction is:

`s_hat = 1 + sum_t p_t`

The L2 penalty is:

`I(y <= 2) * (max(s_hat - y - margin, 0) / 4)^2`

The final loss is:

`mean_i(L_i_ord + lambda_low * P_i_low)`

The first two variants are:

- L2a: `lambda_low=0.3`, `margin=0.0`
- L2b: `lambda_low=0.5`, `margin=0.0`

If L2 lowers `low_to_high_rate` but hurts high-score metrics such as `Acc@5`
or `high_to_mid_or_low_rate`, the follow-up L4 experiment should test high-score
preservation.

## L3b Weighted Threshold-Aligned Low-Score Loss

L3b tests the diagnosis that the expected-score L2 penalty was not well aligned
with `low_to_high_rate`. It reuses the L1 train-split class weights for the base
ordinal BCE and adds a direct low-score threshold suppression term. It does not
use the L2 expected-score penalty.

For low-score samples only:

`P_thr = I(y <= 2) * (p_gt_3^2 + p_gt_4^2) / 2`

The final loss is:

`sum_i(w_i * L_i_ord) / sum_i(w_i) + mu_thr * mean_i(P_i_thr)`

The first L3b variant is:

- L3b: `mu_thr=0.3`

L3a, L4, synthetic data, and calibration are intentionally not implemented in
this step.

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
| L1_weighted_ordinal | completed | 0.7250 | 0.3894 | 0.3504 | 0.7132 | 0.6149 | 0.2136 |
| L2a_asymmetric_ordinal_lambda03_margin0 | completed | 0.7160 | 0.4080 | 0.3705 | 0.6356 | 0.5991 | 0.3398 |
| L2b_asymmetric_ordinal_lambda05_margin0 | completed | 0.7164 | 0.3971 | 0.3633 | 0.6745 | 0.6038 | 0.2718 |
| L3b_weighted_threshold_mu03 | completed | 0.7205 | 0.3946 | 0.3602 | 0.6871 | 0.6122 | 0.2330 |

## Output Files

- Summary table: `thesis_exp/outputs/exp05_low_score_loss/tables/loss_ablation_summary.csv`
- Low-score table: `thesis_exp/outputs/exp05_low_score_loss/tables/loss_ablation_low_score.csv`
- Per-bin table: `thesis_exp/outputs/exp05_low_score_loss/tables/loss_ablation_per_bin.csv`
- Delta table: `thesis_exp/outputs/exp05_low_score_loss/tables/loss_ablation_delta_vs_L0.csv`
- Delta vs L1 table: `thesis_exp/outputs/exp05_low_score_loss/tables/loss_ablation_delta_vs_L1.csv`
- Delta vs L2b table: `thesis_exp/outputs/exp05_low_score_loss/tables/loss_ablation_delta_vs_L2b.csv`
- Tradeoff table: `thesis_exp/outputs/exp05_low_score_loss/tables/loss_ablation_tradeoff.csv`
