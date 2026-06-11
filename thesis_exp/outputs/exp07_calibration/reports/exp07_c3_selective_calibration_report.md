# Exp7-C3 Selective Risk-aware Calibration Report

Formal status: completed.

Training run: NO.

API calls: NO.

Synthetic data generation: NO.

Selection protocol: rejection policy and review budget are selected on dev only; test is used once
for final evaluation of the dev-selected policies.

## Answer Summary

- Selective calibration outperforms full-coverage calibration on the target risk metric: YES for low-to-high risk among non-rejected samples.
- Best coverage>=80% policy: QD-B1_human_only_L1_weighted_ordinal / raw / entropy_confidence_risk at review budget 0.20.
- Test review rate: 0.1995; coverage: 0.8005.
- Test low_to_high: 0.4516 full-coverage -> 0.3200 non-rejected.
- Rejection captures 6/14 potential low_to_high errors (0.4286).
- Thesis suitability: YES, as a human-in-the-loop selective scoring method.
- Framing: this is not a pure automatic scorer; it is automatic scoring with a targeted human-review option for high-risk cases.

## Test Results For Dev-selected Policies

| base | method | policy | review | coverage | low_to_high full -> kept | MAE full -> kept | QWK full -> kept | Acc@5 full -> kept | capture |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QD-B0_human_only_ordinary_ordinal | raw | low_score_overestimation_risk | 0.1995 | 0.8005 | 0.5161 -> 0.5000 | 0.4019 -> 0.3994 | 0.5976 -> 0.6321 | 0.7873 -> 0.7372 | 0.0625 |
| QD-B0_human_only_ordinary_ordinal | risk_aware_threshold | combined_risk | 0.1995 | 0.8005 | 0.5161 -> 0.4444 | 0.4022 -> 0.3884 | 0.6030 -> 0.6479 | 0.7695 -> 0.7484 | 0.2500 |
| QD-B1_human_only_L1_weighted_ordinal **selected** | raw | entropy_confidence_risk | 0.1995 | 0.8005 | 0.4516 -> 0.3200 | 0.4279 -> 0.3783 | 0.6012 -> 0.6649 | 0.7419 -> 0.8125 | 0.4286 |
| QD-B1_human_only_L1_weighted_ordinal | risk_aware_threshold | combined_risk | 0.1995 | 0.8005 | 0.5484 -> 0.5333 | 0.4213 -> 0.4198 | 0.5886 -> 0.6130 | 0.7419 -> 0.7061 | 0.0588 |
| QD-R1_CORAL_human_only | raw | combined_risk | 0.1995 | 0.8005 | 0.5161 -> 0.4231 | 0.4467 -> 0.4062 | 0.5272 -> 0.6008 | 0.8977 -> 0.9455 | 0.3125 |
| QD-R1_CORAL_human_only | risk_aware_threshold | combined_risk | 0.1995 | 0.8005 | 0.4839 -> 0.3333 | 0.4358 -> 0.4009 | 0.5697 -> 0.6273 | 0.8653 -> 0.9091 | 0.4667 |

## Selection Notes

- Dev candidates evaluated: 96.
- Dev-selected base/method policies: 6.
- Success criteria: coverage >= 0.80, low_to_high_non_rejected <= 0.35, review_rate <= 0.20, and Acc@5 within 0.05 of the corresponding full-coverage model.
- Exp7-B gate: remains blocked; C3 is a post-hoc human-review option, not evidence for starting more training.
