# Exp5 L2 Diagnosis Report

This diagnosis reads the completed Exp5 L2a/L2b run outputs only. It does not
train models, rewrite completed experiment outputs, or use checkpoint/weight files.

## Summary

L2 did not meet its main goal. Both L2 variants increased test `low_to_high_rate`
relative to L1, and both hurt overall quality. L2b (`lambda_low=0.5`) is less bad
than L2a (`lambda_low=0.3`), which suggests the asymmetric term has some effect,
but the effect is not strong or aligned enough to beat the weighted L1 baseline.

| method | status | Accuracy | MAE_label | QWK | Kendall | low_to_high | low_bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L0 Exp4 O3 ordinal | completed | 0.7381 | 0.3777 | 0.7036 | 0.6238 | 0.2330 | 1.1812 |
| L1 weighted ordinal | completed | 0.7250 | 0.3894 | 0.7132 | 0.6149 | 0.2136 | 1.1230 |
| L2a lambda=0.3 margin=0 | completed | 0.7160 | 0.4080 | 0.6356 | 0.5991 | 0.3398 | 1.5793 |
| L2b lambda=0.5 margin=0 | completed | 0.7164 | 0.3971 | 0.6745 | 0.6038 | 0.2718 | 1.2977 |

## Checkpoint Selection

The trainer selected checkpoints by `dev_MAE_label`. This is implemented correctly:
for both L2 runs, the selected checkpoint is the best `dev_MAE_label` epoch. However,
L2a has a metric mismatch: the selected epoch is not the best dev `low_to_high_rate`
epoch. L2b is selected at an epoch tied for the best dev `low_to_high_rate` value, so
checkpoint selection mismatch is not sufficient to explain the L2b failure.

| run_id | selected_epoch | best_dev_MAE_epoch | best_dev_low_to_high_epochs | selected_matches_MAE | selected_has_best_low_to_high_value | selected_low_to_high | best_low_to_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L2a_asymmetric_ordinal_lambda03_margin0 | 8 | 8 | 7 | True | False | 0.3000 | 0.2500 |
| L2b_asymmetric_ordinal_lambda05_margin0 | 10 | 10 | 7,8,9,10 | True | True | 0.2500 | 0.2500 |

## Penalty Activation

The L2 penalty is active, but sparse and numerically small. The training set has only
76 low-score examples out of 2654 (`label_5 <= 2`, 2.86%). The logged debug rows show
active low-score penalty in about 11.4% of optimizer-step logs. Because the trainer
logs the latest micro-batch debug at each optimizer step, these rates should be read
as a conservative debug signal rather than exact exposure counts.

| run_id | active_step_rate | mean_penalty_all | lambda_x_penalty | epoch1_penalty | epoch10_penalty | s_hat_active_epoch1 | s_hat_active_epoch10 | penalty_decreased | s_hat_active_decreased |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2a_asymmetric_ordinal_lambda03_margin0 | 0.1143 | 0.003105 | 0.000932 | 0.003721 | 0.001193 | 4.2364 | 2.3173 | True | True |
| L2b_asymmetric_ordinal_lambda05_margin0 | 0.1143 | 0.002944 | 0.001472 | 0.003666 | 0.000635 | 4.2198 | 2.1535 | True | True |

The active low-score `s_hat` drops substantially across training, so the penalty is
not dead. But the mean penalty contribution is tiny after multiplying by `lambda_low`,
and low-score samples remain rare enough that the ordinary ordinal loss is still
mostly shaped by labels 4 and 5.

## Low-Score Prediction Distribution

For true low-score test examples (`label_5 <= 2`), L2b moves some mass back toward
labels 1/2 compared with L2a, but it still predicts label 4/5 too often. The current
expected-score penalty is only indirectly related to the actual risk metric:
`low_to_high_rate` counts low examples with `pred_label >= 4`, which is driven most
directly by the `gt_3`/`gt_4` threshold decisions.

| method | low_n | pred1 | pred2 | pred3 | pred4 | pred5 | low_to_high | mean_p_gt_3 | mean_p_gt_4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0 Exp4 O3 ordinal | 103 | 16 (0.1553) | 31 (0.3010) | 32 (0.3107) | 17 (0.1650) | 7 (0.0680) | 0.2330 | 0.3158 | 0.1286 |
| L1 weighted ordinal | 103 | 20 (0.1942) | 28 (0.2718) | 33 (0.3204) | 14 (0.1359) | 8 (0.0777) | 0.2136 | 0.2500 | 0.1577 |
| L2a lambda=0.3 margin=0 | 103 | 7 (0.0680) | 26 (0.2524) | 35 (0.3398) | 21 (0.2039) | 14 (0.1359) | 0.3398 | 0.3985 | 0.1897 |
| L2b lambda=0.5 margin=0 | 103 | 18 (0.1748) | 22 (0.2136) | 35 (0.3398) | 18 (0.1748) | 10 (0.0971) | 0.2718 | 0.3534 | 0.1903 |

## Likely Failure Causes

| cause | diagnosis | evidence |
| --- | --- | --- |
| penalty too weak | Likely yes | Mean penalty is around 0.003 before multiplying by lambda; low-score examples are 2.86% of training. |
| checkpoint metric mismatch | Partial | L2a selected best MAE epoch but not best low-to-high epoch; L2b does not have this mismatch yet still underperforms L1. |
| expected-score penalty not aligned with low_to_high | Likely yes | The penalty acts on `s_hat`, while low-to-high is thresholded at `pred_label >= 4`, especially gt_3/gt_4. |
| low-score sample variance | Likely yes | Dev has only 20 low-score examples and test has 103, so epoch-level low metrics are coarse and volatile. |
| implementation issue | Not supported | Toy checks pass, selected checkpoints match dev_MAE, loss debug activates, and L2b improves over L2a in the expected direction. |

## Next Step Recommendation

| question | recommendation | rationale | evidence |
| --- | --- | --- | --- |
| proceed_to_L3_weighted_plus_asymmetric | YES_WITH_CAUTION | L2 alone removed class weights and low-score samples are only 2.86% of train; L1 still has the best low_to_high_rate, so combining L1 weighting with a more aligned penalty is the most defensible next ablation. | L1 low_to_high=0.2136, L2b low_to_high=0.2718, train low<=2=76/2654. |
| try_threshold_suppression_penalty | YES | low_to_high_rate is thresholded at pred_label>=4, but the current penalty acts on expected score; threshold-level penalties on gt_3/gt_4 logits for label<=2 align more directly with the metric. | L2b still predicts label>=4 for 27.18% of low test samples; mean p_gt_3/p_gt_4 diagnostics are reported in l2_low_score_prediction_distribution.csv. |
| try_composite_checkpoint_selection | YES_SMALL_SCOPE | L2a selected the best dev_MAE epoch, but not the best dev_low_to_high value; L2b selected an epoch tied for best dev_low_to_high. Composite selection could help but will not solve the main L2b gap vs L1. | L2a selected epoch 8 vs best low_to_high epoch 7. |
| try_larger_lambda_or_margin | LARGER_LAMBDA_ONLY_AS_DIAGNOSTIC; DO_NOT_INCREASE_MARGIN | lambda=0.5 improved over lambda=0.3 but still underperformed L1; margin=0 is already strict, increasing margin would make the penalty weaker. | L2a low_to_high=0.3398, L2b low_to_high=0.2718, L1 low_to_high=0.2136; L2b QWK=0.6745 vs L1 QWK=0.7132. |
| skip_L3_and_move_to_Exp6 | NO_UNLESS_TIME_CONSTRAINED | The negative L2 result is informative but does not isolate whether class weighting plus a better-aligned asymmetric penalty helps. A small L3/threshold-level variant is still scientifically useful. | L2 failure appears driven by low-score sparsity and penalty alignment, not an implementation error. |

## Output Tables

- `thesis_exp/outputs/exp05_low_score_loss/tables/l2_epoch_diagnostics.csv`
- `thesis_exp/outputs/exp05_low_score_loss/tables/l2_penalty_diagnostics.csv`
- `thesis_exp/outputs/exp05_low_score_loss/tables/l2_low_score_prediction_distribution.csv`
- `thesis_exp/outputs/exp05_low_score_loss/tables/l2_next_step_recommendation.csv`
