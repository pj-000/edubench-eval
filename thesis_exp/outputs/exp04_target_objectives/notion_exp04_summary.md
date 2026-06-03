# Exp4 Notion Summary

Question: with A4 input fixed, should 1-5 labels be modeled as classification,
regression, or ordinal classification?

| objective | status | Accuracy | MAE_label | MAE_expected | QWK | Kendall tau | severe_error_rate | low_to_high_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O1_classification | reused_exp03_a4 | 0.7412 | 0.4031 | 0.3666 | 0.6233 | 0.5941 | 0.0428 | 0.4466 |
| O2_regression_smoothl1 | completed | 0.4820 | 0.6162 | 0.5616 | 0.2390 | 0.3652 | 0.0528 | 0.9709 |
| O3_ordinal | completed | 0.5289 | 0.6121 | 0.5646 | 0.0719 | 0.2325 | 0.0586 | 1.0000 |

Recommended reading:

- Use Accuracy for exact label match.
- Use MAE_label and MAE_expected for score-distance errors.
- Use severe_error_rate and low_to_high_rate for educational risk.
