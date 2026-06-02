# Exp2 Review Package

Review status: formal outputs generated and postprocess check passed.

Core files:

- `tables/metrics_summary.csv`
- `tables/per_bin_metrics.csv`
- `tables/low_score_metrics.csv`
- `predictions/predictions_test.jsonl`
- `arrays/exp02_dev_test_arrays.npz`
- `tables/pdf_exp1_baseline_comparison.csv`
- `figures/`

Key test results:

| metric | value |
| --- | ---: |
| Accuracy | 0.7299 |
| MAE_label | 0.4238 |
| MAE_expected | 0.3865 |
| Signed Bias label | +0.1410 |
| Kendall tau | 0.5693 |
| Spearman rho | 0.6361 |
| Acc@1 | 0.1786 |
| Acc@2 | 0.2553 |
| Acc@5 | 0.8085 |
| low_to_high_rate | 0.5340 |

Interpretation:

Exp2 is competitive overall but not solved. Its low-score overestimation rate
remains high, so downstream experiments should not optimize only average MAE or
accuracy.
