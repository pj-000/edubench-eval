# Exp12 Readability Check

Status: `PASS`

| check | status | details |
| --- | --- | --- |
| py_compile thesis_exp/src/edujudge/exp09_pairwise_ordinal/train_qdpr2_anchored_pairwise.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp09_pairwise_ordinal/losses.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp12_monotonic_projection_map_oc/__init__.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp12_monotonic_projection_map_oc/monotone_projection.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp12_monotonic_projection_map_oc/test_monotone_projection.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp12_monotonic_projection_map_oc/collect_exp12_results.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp12_monotonic_projection_map_oc/preflight_exp12.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp12_monotonic_projection_map_oc/readability_check_exp12.py | PASS |  |
| bash -n thesis_exp/scripts/run_exp12_monotonic_projection_map_oc.sh | PASS |  |
| bash -n thesis_exp/scripts/run_exp12_monotonic_projection_map_oc_smoke.sh | PASS |  |
| bash -n thesis_exp/scripts/sync_exp12_monotonic_projection_map_oc_to_server.sh | PASS |  |
| required table exp12a_decode_projection_metrics.csv | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12a_decode_projection_metrics.csv |
| CSV readable exp12a_decode_projection_metrics.csv | PASS |  |
| required table exp12a_raw_vs_projected_selected.csv | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12a_raw_vs_projected_selected.csv |
| CSV readable exp12a_raw_vs_projected_selected.csv | PASS |  |
| required table exp12a_projection_effect_by_seed_epoch.csv | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12a_projection_effect_by_seed_epoch.csv |
| CSV readable exp12a_projection_effect_by_seed_epoch.csv | PASS |  |
| required table exp12a_low_score_projection_distribution.csv | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12a_low_score_projection_distribution.csv |
| CSV readable exp12a_low_score_projection_distribution.csv | PASS |  |
| required table exp12a_monotonic_by_threshold.csv | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12a_monotonic_by_threshold.csv |
| CSV readable exp12a_monotonic_by_threshold.csv | PASS |  |
| required table exp12b_train_metrics_dev.csv | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12b_train_metrics_dev.csv |
| CSV readable exp12b_train_metrics_dev.csv | PASS |  |
| required table exp12b_train_metrics_test_diagnostic.csv | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12b_train_metrics_test_diagnostic.csv |
| CSV readable exp12b_train_metrics_test_diagnostic.csv | PASS |  |
| required table exp12b_selected_checkpoint_test_metrics.csv | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12b_selected_checkpoint_test_metrics.csv |
| CSV readable exp12b_selected_checkpoint_test_metrics.csv | PASS |  |
| required table exp12b_ablation_summary.csv | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12b_ablation_summary.csv |
| CSV readable exp12b_ablation_summary.csv | PASS |  |
| required table exp12b_raw_vs_projected_metrics.csv | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12b_raw_vs_projected_metrics.csv |
| CSV readable exp12b_raw_vs_projected_metrics.csv | PASS |  |
| required table exp12b_low_to_high_by_label.csv | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12b_low_to_high_by_label.csv |
| CSV readable exp12b_low_to_high_by_label.csv | PASS |  |
| required table exp12b_low_score_prediction_distribution.csv | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12b_low_score_prediction_distribution.csv |
| CSV readable exp12b_low_score_prediction_distribution.csv | PASS |  |
| required table exp12b_monotonic_by_threshold.csv | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12b_monotonic_by_threshold.csv |
| CSV readable exp12b_monotonic_by_threshold.csv | PASS |  |
| required table exp12b_projection_delta_summary.csv | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12b_projection_delta_summary.csv |
| CSV readable exp12b_projection_delta_summary.csv | PASS |  |
| required table exp12b_vs_qdb1_qdpr2_exp11.csv | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12b_vs_qdb1_qdpr2_exp11.csv |
| CSV readable exp12b_vs_qdb1_qdpr2_exp11.csv | PASS |  |
| required table exp12_run_config_summary.csv | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12_run_config_summary.csv |
| CSV readable exp12_run_config_summary.csv | PASS |  |
| required table exp12_checkpoint_inventory.csv | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12_checkpoint_inventory.csv |
| CSV readable exp12_checkpoint_inventory.csv | PASS |  |
| required report exp12_monotonic_projection_map_oc_report.md | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/reports/exp12_monotonic_projection_map_oc_report.md |
| required report exp12_monotonic_projection_map_oc_review_package.md | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/reports/exp12_monotonic_projection_map_oc_review_package.md |
| required report exp12_preflight_report.md | PASS | thesis_exp/outputs/exp12_monotonic_projection_map_oc/reports/exp12_preflight_report.md |
| projected monotonic violation fields exist | PASS |  |
| raw/projected decode_mode field exists | PASS |  |
| low-to-high has ratio and count | PASS |  |
| projection delta columns exist | PASS |  |
| uses_test_for_selection is always false | PASS |  |
| report says test metrics are not used for selection | PASS |  |
| report mentions low-to-high | PASS |  |
| report has clear Exp12A status | PASS |  |
| report has clear Exp12B status | PASS |  |
| no fake training result when missing | PASS |  |
| report avoids forbidden overclaim/RL language | PASS | ['proves', 'solves', 'fully eliminates', 'guarantees', 'dpo', 'rlhf', 'reinforcement learning'] |
| no heavy/raw artifacts under tracked Exp12 outputs | PASS | [] |
