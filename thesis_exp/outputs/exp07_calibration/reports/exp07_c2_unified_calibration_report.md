# Exp7-C2 Unified Calibration After Server Artifact Sync

Scope: local decision calibration after syncing server artifacts. No model training, API calls,
synthetic generation, Exp7-B, new training loss, or raw artifact edits were performed.

## Server Artifact Inventory

| base_model | dev_logits_available | test_logits_available | dev_probs_available | test_probs_available | dev_labels_available | test_labels_available | dev_predictions_available | test_predictions_available | calibration_ready |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QD-B0_human_only_ordinary_ordinal | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| QD-B1_human_only_L1_weighted_ordinal | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| QD-R1_CORAL_human_only | yes | yes | yes | yes | yes | yes | yes | yes | yes |

## Unified Test Summary

| base_model | method | MAE_label | QWK | Kendall tau | low_to_high_rate | Acc@5 |
| --- | --- | --- | --- | --- | --- | --- |
| QD-B0_human_only_ordinary_ordinal | raw | 0.4019341190692051 | 0.5976217823018422 | 0.5293242185530982 | 0.5161290322580645 | 0.7873376623376623 |
| QD-B0_human_only_ordinary_ordinal | temperature_scaling | 0.4019341190692051 | 0.5976217823018422 | 0.5293242185530982 | 0.5161290322580645 | 0.7873376623376623 |
| QD-B0_human_only_ordinary_ordinal | global_threshold | 0.4276216379570867 | 0.5746063403481168 | 0.5017123733811352 | 0.5161290322580645 | 0.663961038961039 |
| QD-B0_human_only_ordinary_ordinal | risk_aware_threshold | 0.40223632517376845 | 0.6030316181685385 | 0.5308261008355015 | 0.5161290322580645 | 0.7694805194805194 |
| QD-B1_human_only_L1_weighted_ordinal | raw | 0.42792384406165007 | 0.6012356007375432 | 0.5265979428764057 | 0.45161290322580644 | 0.7418831168831169 |
| QD-B1_human_only_L1_weighted_ordinal | temperature_scaling | 0.42792384406165007 | 0.6012356007375432 | 0.5265979428764057 | 0.45161290322580644 | 0.7418831168831169 |
| QD-B1_human_only_L1_weighted_ordinal | global_threshold | 0.42127530976125716 | 0.5886485753073037 | 0.5207742045777525 | 0.5483870967741935 | 0.7418831168831169 |
| QD-B1_human_only_L1_weighted_ordinal | risk_aware_threshold | 0.42127530976125716 | 0.5886485753073037 | 0.5207742045777525 | 0.5483870967741935 | 0.7418831168831169 |
| QD-R1_CORAL_human_only | raw | 0.4466606225445754 | 0.5271754702243676 | 0.45560562684313166 | 0.5161290322580645 | 0.8977272727272727 |
| QD-R1_CORAL_human_only | temperature_scaling | 0.4466606225445754 | 0.5271754702243676 | 0.45560562684313166 | 0.5161290322580645 | 0.8977272727272727 |
| QD-R1_CORAL_human_only | global_threshold | 0.4291326684799032 | 0.5724102588902609 | 0.5045974038970956 | 0.5161290322580645 | 0.6801948051948052 |
| QD-R1_CORAL_human_only | risk_aware_threshold | 0.43578120278029614 | 0.5697080205443874 | 0.48795138632878815 | 0.4838709677419355 | 0.8652597402597403 |

## Required Answers

1. Calibrated base models: QD-B0_human_only_ordinary_ordinal, QD-B1_human_only_L1_weighted_ordinal, QD-R1_CORAL_human_only.
2. Best method per base:
   - QD-B0_human_only_ordinary_ordinal: temperature_scaling (temperature=2.0; no decision-threshold change).
   - QD-B1_human_only_L1_weighted_ordinal: temperature_scaling (temperature=2.0; no decision-threshold change).
   - QD-R1_CORAL_human_only: risk_aware_threshold (theta=0.3, 0.3, 0.55, 0.55).
3. Best overall: `QD-B1_human_only_L1_weighted_ordinal` with `temperature_scaling`.
4. Beats QD-B1 raw low_to_high=0.4516? NO; best low_to_high=0.4516.
5. Trade-off in MAE/QWK/Acc@5:
   - QD-B0_human_only_ordinary_ordinal: low_to_high 0.5161 -> 0.5161; MAE 0.4019 -> 0.4019; QWK 0.5976 -> 0.5976; Acc@5 0.7873 -> 0.7873.
   - QD-B1_human_only_L1_weighted_ordinal: low_to_high 0.4516 -> 0.4516; MAE 0.4279 -> 0.4279; QWK 0.6012 -> 0.6012; Acc@5 0.7419 -> 0.7419.
   - QD-R1_CORAL_human_only: low_to_high 0.5161 -> 0.4839; MAE 0.4467 -> 0.4358; QWK 0.5272 -> 0.5697; Acc@5 0.8977 -> 0.8653.
6. Optional rejection analysis:
| base_model | review_rate | coverage | low_to_high_all | low_to_high_non_rejected | Acc@5_all | Acc@5_non_rejected |
| --- | --- | --- | --- | --- | --- | --- |
| QD-B0_human_only_ordinary_ordinal | 0.056210335448776065 | 0.9437896645512239 | 0.5161290322580645 | 0.4827586206896552 | 0.7694805194805194 | 0.7789115646258503 |
| QD-B1_human_only_L1_weighted_ordinal | 0.04533091568449683 | 0.9546690843155031 | 0.5483870967741935 | 0.5333333333333333 | 0.7418831168831169 | 0.7474402730375427 |
| QD-R1_CORAL_human_only | 0.1541251133272892 | 0.8458748866727108 | 0.4838709677419355 | 0.3333333333333333 | 0.8652597402597403 | 0.9 |
7. Exp7-C thesis final method? NOT YET under this unified synced-artifact run.
8. Exp7-B should remain blocked until calibration trade-offs are accepted and documented.

## Method Notes

- QD-B0_human_only_ordinary_ordinal temperature scaling: low_to_high delta=0.0000, MAE delta=0.0000.
- QD-B0_human_only_ordinary_ordinal global threshold: low_to_high delta=0.0000, MAE delta=0.0257.
- QD-B0_human_only_ordinary_ordinal risk-aware threshold: low_to_high delta=0.0000, MAE delta=0.0003.
- QD-B1_human_only_L1_weighted_ordinal temperature scaling: low_to_high delta=0.0000, MAE delta=0.0000.
- QD-B1_human_only_L1_weighted_ordinal global threshold: low_to_high delta=0.0968, MAE delta=-0.0066.
- QD-B1_human_only_L1_weighted_ordinal risk-aware threshold: low_to_high delta=0.0968, MAE delta=-0.0066.
- QD-R1_CORAL_human_only temperature scaling: low_to_high delta=0.0000, MAE delta=0.0000.
- QD-R1_CORAL_human_only global threshold: low_to_high delta=0.0000, MAE delta=-0.0175.
- QD-R1_CORAL_human_only risk-aware threshold: low_to_high delta=-0.0323, MAE delta=-0.0109.
