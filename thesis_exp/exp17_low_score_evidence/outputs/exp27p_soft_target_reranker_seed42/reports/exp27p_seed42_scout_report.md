# Exp27P Seed42 Scout

- status: `PASS`
- recommend seeds 43/44: `true`
- recommend v3_safe16: `true`
- test accessed: `false`

| variant | epoch | MAE | Bias | Exact | Kendall tau | Bin agree | QWK | low-to-high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0_original_unweighted | 3 | 0.3930 | 0.2538 | 0.6902 | 0.5344 | 0.8681 | 0.5455 | 0.5614 |
| v1_original_label_matched_weight | 5 | 0.4029 | 0.1951 | 0.6757 | 0.5100 | 0.8663 | 0.5472 | 0.5614 |
| v2_selective_hard_relabel | 6 | 0.4047 | 0.1626 | 0.6667 | 0.5006 | 0.8744 | 0.5593 | 0.5088 |
| v3_selective_soft_audit | 3 | 0.3957 | 0.3035 | 0.6883 | 0.5445 | 0.8717 | 0.5431 | 0.5789 |
