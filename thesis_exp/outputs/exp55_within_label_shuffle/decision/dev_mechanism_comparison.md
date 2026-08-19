# Within-label shuffled-soft development-set mechanism control

All values are means over paired seeds 42/43/44. No test data were accessed.

| Metric | Hard-only | Shuffled-soft | HMSA | HMSA − Shuffle |
| --- | ---: | ---: | ---: | ---: |
| MAE_human_mean | 0.393072 | 0.389224 | 0.379016 | -0.010207 |
| Exact_rounded | 0.719378 | 0.722390 | 0.730924 | +0.008534 |
| Kendall_human_mean | 0.568976 | 0.576606 | 0.596599 | +0.019993 |
| Bias_human_mean | 0.150100 | 0.120482 | 0.110944 | -0.009538 |
| QWK_rounded | 0.602210 | 0.635157 | 0.648031 | +0.012873 |
| L2H_count | 11.333333 | 8.333333 | 8.666667 | +0.333333 |

For MAE, absolute bias, and L2H, lower is better; for Exact, Kendall, and QWK, higher is better.
HMSA beats the shuffled control for MAE, Exact, Kendall, bias magnitude, and QWK in the three-seed mean.
The shuffled control has 0.333 fewer mean L2H cases, a one-case total difference across the three runs.
