# Exp27P Seed42 Scout

- status: `PASS`
- recommend seeds 43/44: `true`
- recommend v3_safe16: `true`
- test accessed: `false`

| variant | epoch | MAE | QWK | low-to-high | label2 recall | label5 recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0_original_unweighted | 3 | 0.3930 | 0.5455 | 0.5614 | 0.0000 | 0.8759 |
| v1_original_label_matched_weight | 5 | 0.4029 | 0.5472 | 0.5614 | 0.0000 | 0.8201 |
| v2_selective_hard_relabel | 6 | 0.4047 | 0.5593 | 0.5088 | 0.0000 | 0.7860 |
| v3_selective_soft_audit | 3 | 0.3957 | 0.5431 | 0.5789 | 0.0000 | 0.9173 |
