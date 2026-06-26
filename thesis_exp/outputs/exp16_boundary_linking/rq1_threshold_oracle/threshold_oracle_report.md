# Exp16A-Oracle Threshold Analysis

This diagnostic fixes the learned quality score `s` and changes only thresholding rules. It does not
train a model.

## Summary

| variant | split | method | MAE | QWK | Accuracy | low-to-high | label2 recall |
|---|---|---|---:|---:|---:|---:|---:|
| `qmr` | dev | `boundary_tower_tau` | 0.3993 | 0.5532 | 0.6766 | 29/57 (0.5088) | 0.0000 |
| `qmr` | test | `boundary_tower_tau` | 0.3772 | 0.5380 | 0.6718 | 17/31 (0.5484) | 0.0000 |
| `qmr` | dev | `global_threshold_on_s` | 0.5619 | 0.3747 | 0.5248 | 44/57 (0.7719) | 0.0000 |
| `qmr` | dev | `metric_threshold_on_s` | 0.4959 | 0.4459 | 0.6007 | 40/57 (0.7018) | 0.0000 |
| `qmr` | test | `global_threshold_on_s` | 0.5059 | 0.3702 | 0.5403 | 23/31 (0.7419) | 0.0000 |
| `qmr` | test | `metric_threshold_on_s` | 0.4669 | 0.4596 | 0.5966 | 20/31 (0.6452) | 0.0909 |
| `qmr` | dev | `oracle_dev_threshold_on_s` | 0.5565 | 0.3755 | 0.5230 | 44/57 (0.7719) | 0.0000 |
| `metric_rubric` | dev | `boundary_tower_tau` | 0.4237 | 0.5274 | 0.6531 | 33/57 (0.5789) | 0.0000 |
| `metric_rubric` | test | `boundary_tower_tau` | 0.3708 | 0.5524 | 0.6736 | 15/31 (0.4839) | 0.0000 |
| `metric_rubric` | dev | `global_threshold_on_s` | 0.5266 | 0.3667 | 0.5664 | 44/57 (0.7719) | 0.0000 |
| `metric_rubric` | dev | `metric_threshold_on_s` | 0.4363 | 0.4799 | 0.6486 | 44/57 (0.7719) | 0.0000 |
| `metric_rubric` | test | `global_threshold_on_s` | 0.5113 | 0.4009 | 0.5240 | 22/31 (0.7097) | 0.0000 |
| `metric_rubric` | test | `metric_threshold_on_s` | 0.4152 | 0.4876 | 0.6301 | 21/31 (0.6774) | 0.0000 |
| `metric_rubric` | dev | `oracle_dev_threshold_on_s` | 0.5014 | 0.3955 | 0.6043 | 44/57 (0.7719) | 0.0000 |

## Interpretation Guide

- If global or metric thresholds improve label2 recall while MAE/QWK remain similar, the main weakness is tau calibration.
- If threshold calibration cannot recover label2 recall, the learned quality score `s` is not separating label-2 answers from higher-score answers.
- `oracle_dev_threshold_on_s` is a diagnostic upper bound only; it is not a valid model-selection result.

## RQ1 Recommendation

Use this table with the boundary-failure diagnosis: if fixed-s calibration helps, RQ1 should close
with a calibration/boundary story; if it does not help, close RQ1 by noting that boundary generation
alone is insufficient and move to RQ2 risk-aware learning.
