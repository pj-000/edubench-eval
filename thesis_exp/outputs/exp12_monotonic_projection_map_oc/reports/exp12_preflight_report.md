# Exp12 Monotonic Projection / MAP-OC Preflight

Status: `PASS`

| item | status | value | details |
| --- | --- | --- | --- |
| repo_root | PASS | `/home/jpang/edubench-eval-exp2` |  |
| current_git_commit | PASS | `4e95b0a4f1b41a0e7c43bbaf004e8561d136dc68` |  |
| output_dir | PASS | `thesis_exp/outputs/exp12_monotonic_projection_map_oc` |  |
| local_run_root | PASS | `thesis_exp/runs/exp12_monotonic_projection_map_oc` |  |
| local_checkpoint_dir_ignored | PASS | `thesis_exp/runs/exp12_monotonic_projection_map_oc` | thesis_exp/.gitignore ignores runs/ |
| train_data_path | PASS | `thesis_exp/outputs/exp06_question_disjoint_baselines/datasets/A4_question_answer_metric_rubric_metadata_question_seed42/train.jsonl` |  |
| dev_data_path | PASS | `thesis_exp/outputs/exp06_question_disjoint_baselines/datasets/A4_question_answer_metric_rubric_metadata_question_seed42/dev.jsonl` |  |
| test_data_path | PASS | `thesis_exp/outputs/exp06_question_disjoint_baselines/datasets/A4_question_answer_metric_rubric_metadata_question_seed42/test.jsonl` |  |
| QD_B1_checkpoint_path | PASS | `thesis_exp/artifacts/exp06_question_disjoint_baselines/checkpoints/QD-B1_human_only_L1_weighted_ordinal/best` | requires state_dict.pt, exp05_head_metadata.json, tokenizer.json |
| Exp11_outputs_exist | PASS | `thesis_exp/outputs/exp11_checkpoint_selection_sensitivity/tables` |  |
| Exp11_selected_checkpoint_info_exists | PASS | `mae_guard_p_gt_3_low_mean` | Exp12A needs Exp11 selected epoch metadata. |
| model_name | PASS | `/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B` |  |
| seed_list | PASS | `42` |  |
| epochs | PASS | `3` |  |
| projection_method | PASS | `pava` |  |
| projection_in_decode | PASS | `True` |  |
| projection_in_pair_score | PASS | `True` |  |
| projection_in_point_loss | PASS | `True` |  |
| projection_in_anchor | PASS | `True` |  |
| selection_rule | PASS | `mae_guard_p_gt_3_low_mean` |  |
| selection_delta | PASS | `0.005` |  |
| gamma | PASS | `4.0` |  |
| lambda_point | PASS | `1.0` |  |
| lambda_pair | PASS | `0.05` |  |
| lambda_anchor | PASS | `0.5` |  |
| lambda_mono | PASS | `0.1` |  |
| eta_proj | PASS | `0.1` |  |
| gpu_assignment | PASS | `{'42': '6'}` |  |
| formal_run_has_no_sample_caps | PASS | `{'MAX_TRAIN_SAMPLES': '', 'MAX_EVAL_SAMPLES': '', 'MAX_TRAIN_PAIRS': '', 'MAX_DEV_PAIRS': ''}` | Formal Exp12 must not use sample or pair caps. |
| seed_42_Exp11_checkpoint_or_prediction_cache | PASS | `True` | Exp12A can regenerate raw probabilities from Exp11 local checkpoint on server. |
