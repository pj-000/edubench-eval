# Exp4 Target Objective Comparison

Exp4 fixes the Exp3 A4 input template and compares how the 1-5 educational
score label should be modeled during training.

Exp4 is not another input ablation. The input text is fixed to:

`question + answer + metric + rubric + metadata`

The compared objectives are:

- O1 classification: reuse Exp3 A4 5-class CE result.
- O2 regression: train one continuous-score head with SmoothL1Loss on `human_mean_5`.
- O3 ordinal classification: train four threshold logits with BCEWithLogitsLoss.

## Main Results

| objective | status | Accuracy | MAE_label | MAE_expected | QWK | Kendall tau | severe_error_rate | low_to_high_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O1_classification | reused_exp03_a4 | 0.7412 | 0.4031 | 0.3666 | 0.6233 | 0.5941 | 0.0428 | 0.4466 |
| O2_regression_smoothl1 | completed | 0.6984 | 0.3889 | 0.3275 | 0.7127 | 0.6212 | 0.0207 | 0.1553 |
| O3_ordinal | completed | 0.7381 | 0.3777 | 0.3430 | 0.7036 | 0.6238 | 0.0316 | 0.2330 |

## Current Best

- Best MAE_label: O3_ordinal (0.3777)
- Best MAE_expected: O2_regression_smoothl1 (0.3275)
- Best Accuracy: O1_classification (0.7412)
- Lowest severe_error_rate: O2_regression_smoothl1 (0.0207)
- Lowest low_to_high_rate: O2_regression_smoothl1 (0.1553)

## Interpretation Guide

Classification treats 1, 2, 3, 4, and 5 as unrelated classes. It is a strong
baseline but does not encode that confusing 2 with 3 is less severe than
confusing 2 with 5.

Regression treats the target as a continuous human mean score. The first round
uses SmoothL1Loss because it is less dominated by outlier residuals than MSE
while still optimizing distance in score space.

Ordinal classification treats the score as an ordered level. It predicts four
threshold events: score greater than 1, 2, 3, and 4. The first round does not
force monotonic correction; it reports monotonic violation rate so that a
follow-up correction is only added if violations are material.

## Output Files

- Summary table: `thesis_exp/outputs/exp04_target_objectives/tables/target_objective_summary.csv`
- Low-score table: `thesis_exp/outputs/exp04_target_objectives/tables/target_objective_low_score.csv`
- Per-bin table: `thesis_exp/outputs/exp04_target_objectives/tables/target_objective_per_bin.csv`
- Prediction copies: `thesis_exp/outputs/exp04_target_objectives/predictions`
- Array copies: `thesis_exp/outputs/exp04_target_objectives/arrays`
