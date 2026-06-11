# Exp7-C Risk-aware Ordinal Calibration

Scope: calibration only. No model training, API calls, synthetic generation, Exp7-B, new training
loss, or raw artifact edits were performed.

## Base Inventory

| base_model | dev_probs_available | test_probs_available | dev_logits_available | test_logits_available | can_threshold_calibrate | can_temperature_calibrate | blocking_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QD-B0_human_only_ordinary_ordinal | server_ready_local_missing | server_ready_local_missing | server_ready_local_missing | server_ready_local_missing | no | no | server_ready_local_missing; sync artifacts locally or run calibration on the server |
| QD-B1_human_only_L1_weighted_ordinal | server_ready_local_missing | server_ready_local_missing | server_ready_local_missing | server_ready_local_missing | no | no | server_ready_local_missing; sync artifacts locally or run calibration on the server |
| QD-R1_CORAL_human_only | yes_local | yes_local | yes_local | yes_local | yes | yes | ready |

## Test Summary

| base_model | method | split | MAE_label | QWK | Kendall tau | low_to_high_rate | Acc@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QD-R1_CORAL_human_only | raw | test | 0.4466606225445754 | 0.5271754702243676 | 0.45560562684313166 | 0.5161290322580645 | 0.8977272727272727 |
| QD-R1_CORAL_human_only | temperature_scaling | test | 0.4466606225445754 | 0.5271754702243676 | 0.45560562684313166 | 0.5161290322580645 | 0.8977272727272727 |
| QD-R1_CORAL_human_only | global_threshold | test | 0.4291326684799032 | 0.5724102588902609 | 0.5045974038970956 | 0.5161290322580645 | 0.6801948051948052 |
| QD-R1_CORAL_human_only | risk_aware_threshold | test | 0.43578120278029614 | 0.5697080205443874 | 0.48795138632878815 | 0.4838709677419355 | 0.8652597402597403 |
| QD-B0_human_only_ordinary_ordinal | raw_reference_existing_summary | test | 0.4019341190692051 | 0.5976217823018422 | 0.5293242185530982 | 0.5161290322580645 | 0.7873376623376623 |
| QD-B1_human_only_L1_weighted_ordinal | raw_reference_existing_summary | test | 0.42792384406165007 | 0.6012356007375432 | 0.5265979428764057 | 0.45161290322580644 | 0.7418831168831169 |

## Required Answers

1. Calibration-ready bases: QD-R1_CORAL_human_only.
2. Temperature scaling help? NO; low_to_high delta=0.0000, MAE delta=0.0000 on QD-R1_CORAL_human_only.
3. Global threshold calibration help? NO; low_to_high delta=0.0000, MAE delta=-0.0175 on QD-R1_CORAL_human_only.
4. Risk-aware threshold calibration reduce low_to_high? YES; low_to_high delta=-0.0323, MAE delta=-0.0109 on QD-R1_CORAL_human_only.
5. Trade-off: best method `risk_aware_threshold` changes low_to_high 0.5161 -> 0.4839, MAE 0.4467 -> 0.4358, QWK 0.5272 -> 0.5697, Acc@5 0.8977 -> 0.8653.
6. Best calibrated model: `QD-R1_CORAL_human_only` with `risk_aware_threshold`.
7. Calibration outperform QD-B1 raw? NO; best low_to_high=0.4839, QD-B1 raw low_to_high=0.4516.
8. Calibration addresses low-score overestimation without additional training or synthetic data; this run is a decision-layer intervention only.
9. Exp7-C final method? NOT YET based on current local artifacts.
10. Limitations: only bases with local logits/probs can be calibrated here; B0/B1 require artifact sync or server-side calibration.

## Selected Risk-aware Config

| base_model | lambda_low | eta_high | theta_1 | theta_2 | theta_3 | theta_4 | dev_objective | selection_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QD-R1_CORAL_human_only | 0.5 | 0.5 | 0.3 | 0.3 | 0.55 | 0.55 | 0.7693581691486063 | best_constrained |

## Optional Rejection Analysis

| base_model | review_rate | coverage | low_to_high_all | low_to_high_non_rejected | Acc@5_all | Acc@5_non_rejected |
| --- | --- | --- | --- | --- | --- | --- |
| QD-R1_CORAL_human_only | 0.1541251133272892 | 0.8458748866727108 | 0.4838709677419355 | 0.3333333333333333 | 0.8652597402597403 | 0.9 |
