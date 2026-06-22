# Exp11 Checkpoint Selection Sensitivity Preflight

Status: `PASS`

| item | status | value | details |
| --- | --- | --- | --- |
| repo_root | PASS | `/home/jpang/edubench-eval-exp2` |  |
| current_git_commit | PASS | `4e95b0a4f1b41a0e7c43bbaf004e8561d136dc68` |  |
| output_dir | PASS | `thesis_exp/outputs/exp11_checkpoint_selection_sensitivity` |  |
| local_run_root | PASS | `thesis_exp/runs/exp11_checkpoint_selection_sensitivity` |  |
| local_checkpoint_dir_ignored | PASS | `thesis_exp/runs/exp11_checkpoint_selection_sensitivity` | thesis_exp/.gitignore ignores runs/ |
| train_data_path | PASS | `thesis_exp/outputs/exp06_question_disjoint_baselines/datasets/A4_question_answer_metric_rubric_metadata_question_seed42/train.jsonl` |  |
| dev_data_path | PASS | `thesis_exp/outputs/exp06_question_disjoint_baselines/datasets/A4_question_answer_metric_rubric_metadata_question_seed42/dev.jsonl` |  |
| test_data_path | PASS | `thesis_exp/outputs/exp06_question_disjoint_baselines/datasets/A4_question_answer_metric_rubric_metadata_question_seed42/test.jsonl` |  |
| QD_B1_checkpoint_path | PASS | `thesis_exp/artifacts/exp06_question_disjoint_baselines/checkpoints/QD-B1_human_only_L1_weighted_ordinal/best` | requires state_dict.pt, exp05_head_metadata.json, tokenizer.json |
| model_name | PASS | `/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B` |  |
| seed_list | PASS | `42 43 44` |  |
| epochs | PASS | `3` |  |
| lambda_point | PASS | `1.0` |  |
| lambda_pair | PASS | `0.05` |  |
| lambda_anchor | PASS | `0.5` |  |
| lambda_mono | PASS | `0.1` |  |
| pair_dataset_size_train | PASS | `10000` |  |
| pair_dataset_size_dev | PASS | `3000` |  |
| max_pairs_per_record | PASS | `80` |  |
| max_pairs_per_low_record | PASS | `100` |  |
| selection_rules | PASS | `dev_mae_min, dev_qwk_max, dev_low_to_high_min_diagnostic, mae_guard_soft_risk, mae_guard_label2_soft_risk, mae_guard_p_gt_3_low_mean, mae_guard_soft_risk_mono, last_epoch_diagnostic` |  |
| gamma | PASS | `4.0` |  |
| delta | PASS | `0.005` |  |
| beta | PASS | `0.2` |  |
| gpu_assignment | PASS | `{'42': '6', '43': '7', '44': '6'}` |  |
| formal_run_has_no_sample_caps | PASS | `{'MAX_TRAIN_SAMPLES': '', 'MAX_EVAL_SAMPLES': '', 'MAX_TRAIN_PAIRS': '', 'MAX_DEV_PAIRS': ''}` | Formal Exp11 must not use sample or pair caps. |
| seed_42_run_dir | PASS | `thesis_exp/runs/exp11_checkpoint_selection_sensitivity/seed_42/run` |  |
| seed_42_checkpoint_dir | PASS | `thesis_exp/runs/exp11_checkpoint_selection_sensitivity/seed_42/checkpoints` | local ignored checkpoint directory |
| seed_43_run_dir | PASS | `thesis_exp/runs/exp11_checkpoint_selection_sensitivity/seed_43/run` |  |
| seed_43_checkpoint_dir | PASS | `thesis_exp/runs/exp11_checkpoint_selection_sensitivity/seed_43/checkpoints` | local ignored checkpoint directory |
| seed_44_run_dir | PASS | `thesis_exp/runs/exp11_checkpoint_selection_sensitivity/seed_44/run` |  |
| seed_44_checkpoint_dir | PASS | `thesis_exp/runs/exp11_checkpoint_selection_sensitivity/seed_44/checkpoints` | local ignored checkpoint directory |
