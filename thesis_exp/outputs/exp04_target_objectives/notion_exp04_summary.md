# Exp4 Notion Summary

Question: with A4 input fixed, should 1-5 labels be modeled as classification,
regression, or ordinal classification?

| objective | status | Accuracy | MAE_label | MAE_expected | QWK | Kendall tau | severe_error_rate | low_to_high_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O1_classification | reused_exp03_a4 | 0.7412 | 0.4031 | 0.3666 | 0.6233 | 0.5941 | 0.0428 | 0.4466 |
| O2_regression_smoothl1 | completed | 0.6984 | 0.3889 | 0.3275 | 0.7127 | 0.6212 | 0.0207 | 0.1553 |
| O3_ordinal | completed | 0.7381 | 0.3777 | 0.3430 | 0.7036 | 0.6238 | 0.0316 | 0.2330 |

Recommended reading:

- Use Accuracy for exact label match.
- Use MAE_label and MAE_expected for score-distance errors.
- Use severe_error_rate and low_to_high_rate for educational risk.
