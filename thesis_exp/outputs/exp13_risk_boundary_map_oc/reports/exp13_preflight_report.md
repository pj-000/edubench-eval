# Exp13 Risk-Boundary MAP-OC Preflight

Status: `PASS`

| item | status | value | details |
| --- | --- | --- | --- |
| repo_root | PASS | `/home/jpang/edubench-eval-exp2` |  |
| current_git_commit | PASS | `4e95b0a4f1b41a0e7c43bbaf004e8561d136dc68` |  |
| mode | PASS | `scout` |  |
| eval_test | PASS | `0` | scout should be 0; formal should be 1 |
| output_dir | PASS | `thesis_exp/outputs/exp13_risk_boundary_map_oc` |  |
| local_run_root | PASS | `thesis_exp/runs/exp13_risk_boundary_map_oc` |  |
| local_checkpoint_dir_ignored | PASS | `thesis_exp/runs/exp13_risk_boundary_map_oc` | thesis_exp/.gitignore ignores runs/ |
| train_data_path | PASS | `thesis_exp/outputs/exp06_question_disjoint_baselines/datasets/A4_question_answer_metric_rubric_metadata_question_seed42/train.jsonl` |  |
| dev_data_path | PASS | `thesis_exp/outputs/exp06_question_disjoint_baselines/datasets/A4_question_answer_metric_rubric_metadata_question_seed42/dev.jsonl` |  |
| test_data_path | PASS | `thesis_exp/outputs/exp06_question_disjoint_baselines/datasets/A4_question_answer_metric_rubric_metadata_question_seed42/test.jsonl` |  |
| QD_B1_checkpoint_path | PASS | `thesis_exp/artifacts/exp06_question_disjoint_baselines/checkpoints/QD-B1_human_only_L1_weighted_ordinal/best` | requires state_dict.pt, exp05_head_metadata.json, tokenizer.json |
| seed_list | PASS | `42` |  |
| gpu_list | PASS | `6 7` |  |
| run_list | PASS | `point_pair_proj_l2h_lam0p20` |  |
| epochs | PASS | `1` |  |
| selection_delta | PASS | `0.005` |  |
| formal_run_has_no_sample_caps | PASS | `{'MAX_TRAIN_SAMPLES': '', 'MAX_EVAL_SAMPLES': '', 'MAX_TRAIN_PAIRS': '', 'MAX_DEV_PAIRS': ''}` | Formal/scout Exp13 scripts must not use sample or pair caps. |
| config_point_pair_proj_l2h_lam0p20 | PASS | `thesis_exp/configs/exp13_risk_boundary_map_oc/exp13_point_pair_proj_l2h_lam0p20.yaml` |  |
| point_pair_proj_l2h_lam0p20_risk_threshold_t | PASS | `3` | threshold tensor index is 2 in the loss implementation |
| point_pair_proj_l2h_lam0p20_selection_rule | PASS | `mae_guard_p_gt_3_low_mean` |  |
