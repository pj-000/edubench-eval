# Exp13 Formal Lock Report

Status: `PASS`

Formal config was locked from dev-only scout results.
Test metrics are only for final held-out evaluation.
No test labels are used for tuning.

## Locked Contract

- Locked from commit: `8457068`
- Locked mode: `scout`
- Selection source: `thesis_exp/outputs/exp13_risk_boundary_map_oc/tables/exp13_dev_selected_configs.csv`
- Locked runs: `score_proj_l2h_lam0p20`
- Locked seeds: `42, 43, 44`
- Selection rule: `mae_guard_p_gt_3_low_mean`
- Selection delta: `0.005`
- Eval test: `True`

## Checks

| check | status | details |
| --- | --- | --- |
| formal lock YAML exists | PASS | thesis_exp/configs/exp13_risk_boundary_map_oc/exp13_formal_locked.yaml |
| locked run is score_proj_l2h_lam0p20 | PASS | ['score_proj_l2h_lam0p20'] |
| locked seeds are 42/43/44 | PASS | [42, 43, 44] |
| selection source is configured | PASS | thesis_exp/outputs/exp13_risk_boundary_map_oc/tables/exp13_dev_selected_configs.csv |
| selection source CSV exists | PASS | thesis_exp/outputs/exp13_risk_boundary_map_oc/tables/exp13_dev_selected_configs.csv |
| selection source CSV has no test columns | PASS | ['mode', 'run_name', 'seed', 'epoch', 'global_step', 'decode_mode', 'selected', 'eligible', 'selection_rule', 'selection_delta', 'uses_test_for_selection', 'dev_MAE', 'dev_QWK', 'dev_Accuracy', 'dev_Acc@5', 'dev_low_to_high', 'dev_low_to_high_count', 'dev_true_low_score_count', 'dev_label1_low_to_high', 'dev_label1_low_to_high_count', 'dev_label2_low_to_high', 'dev_label2_low_to_high_count', 'dev_monotonic_violation', 'dev_p1_lt_p2', 'dev_p2_lt_p3', 'dev_p3_lt_p4', 'dev_p_gt_3_low_mean', 'dev_p_gt_3_label1_mean', 'dev_p_gt_3_label2_mean', 'dev_p_gt_3_low_q90', 'dev_p_gt_3_low_q95', 'dev_low_count_with_p_gt_3_over_0p5', 'dev_high_to_low', 'dev_high_to_low_count', 'dev_label4_recall', 'dev_label5_recall'] |
| selection source contains locked run | PASS | score_proj_l2h_lam0p20 |
| locked run uses_test_for_selection is false | PASS | ['False'] |
