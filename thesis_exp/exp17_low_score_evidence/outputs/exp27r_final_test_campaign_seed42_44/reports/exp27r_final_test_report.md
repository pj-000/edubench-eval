# Exp27R Final One-Shot Test

- final paper position: `directional_signal_only`
- test access count: `1`
- methods/training/data frozen: `true`
- final test closed: `true`

| variant | MAE | QWK | low-to-high | label2 recall | label5 recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0_original_unweighted | 0.3633 | 0.5584 | 0.5914 | 0.0000 | 0.7749 |
| v1_original_label_matched_weight | 0.3708 | 0.5301 | 0.5806 | 0.0000 | 0.8436 |
| v2_selective_hard_relabel | 0.3753 | 0.5516 | 0.5269 | 0.0000 | 0.7408 |
| v3_selective_soft_audit | 0.3747 | 0.5375 | 0.6129 | 0.0152 | 0.7689 |
| v3_safe16_original_low_anchor | 0.3759 | 0.5605 | 0.5054 | 0.0152 | 0.7776 |
