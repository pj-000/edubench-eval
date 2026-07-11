# Exp27P Multiseed Stability

- status: `SOFT_TARGET_NOT_STABLE`
- seeds: `[42, 43, 44]`
- v3 mean performance gate: `false`
- v3 seedwise gate pass: `1/3`
- recommend v3_safe16: `true`
- test accessed: `false`

| variant | MAE | Bias | Exact | Kendall tau | Bin agree | QWK | low-to-high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0_original_unweighted | 0.4035±0.0145 | 0.1933±0.0683 | 0.6760±0.0136 | 0.5118±0.0264 | 0.8672±0.0009 | 0.5403±0.0096 | 0.5497±0.0101 |
| v1_original_label_matched_weight | 0.4101±0.0083 | 0.2487±0.0501 | 0.6730±0.0033 | 0.5067±0.0074 | 0.8699±0.0048 | 0.5324±0.0171 | 0.5673±0.0442 |
| v2_selective_hard_relabel | 0.4122±0.0146 | 0.1563±0.0254 | 0.6661±0.0063 | 0.4931±0.0161 | 0.8654±0.0080 | 0.5390±0.0244 | 0.5439±0.0608 |
| v3_selective_soft_audit | 0.4065±0.0158 | 0.2120±0.0940 | 0.6748±0.0127 | 0.5092±0.0356 | 0.8711±0.0019 | 0.5312±0.0133 | 0.5673±0.0365 |
