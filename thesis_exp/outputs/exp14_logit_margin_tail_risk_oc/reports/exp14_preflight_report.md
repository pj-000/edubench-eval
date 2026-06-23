# Exp14 Logit-Margin Tail-Risk OC Preflight

Status: `PASS`

| item | status | value | details |
| --- | --- | --- | --- |
| repo_root | PASS | `/home/jpang/edubench-eval-exp2` |  |
| current_git_commit | PASS | `4e95b0a4f1b41a0e7c43bbaf004e8561d136dc68` |  |
| mode | PASS | `scout` | Exp14 currently implements scout-only by default. |
| eval_test | PASS | `0` | Scout must not evaluate test. |
| allow_exp14_test_guard | PASS | `0` | EVAL_TEST=1 requires ALLOW_EXP14_TEST=1. |
| output_dir | PASS | `thesis_exp/outputs/exp14_logit_margin_tail_risk_oc` |  |
| local_run_root | PASS | `thesis_exp/runs/exp14_logit_margin_tail_risk_oc` |  |
| local_checkpoint_dir_ignored | PASS | `thesis_exp/runs/exp14_logit_margin_tail_risk_oc` | thesis_exp/.gitignore ignores runs/ |
| train_data_path | PASS | `thesis_exp/outputs/exp06_question_disjoint_baselines/datasets/A4_question_answer_metric_rubric_metadata_question_seed42/train.jsonl` |  |
| dev_data_path | PASS | `thesis_exp/outputs/exp06_question_disjoint_baselines/datasets/A4_question_answer_metric_rubric_metadata_question_seed42/dev.jsonl` |  |
| test_data_path | PASS | `thesis_exp/outputs/exp06_question_disjoint_baselines/datasets/A4_question_answer_metric_rubric_metadata_question_seed42/test.jsonl` | present but not used in scout |
| QD_B1_checkpoint_path | PASS | `thesis_exp/artifacts/exp06_question_disjoint_baselines/checkpoints/QD-B1_human_only_L1_weighted_ordinal/best` | requires state_dict.pt, exp05_head_metadata.json, tokenizer.json |
| seed_list | PASS | `42` |  |
| gpu_list | PASS | `6 7` |  |
| run_list | PASS | `score_logit_margin_lam0p01_alllow score_logit_margin_lam0p02_alllow score_logit_margin_lam0p05_alllow score_tail_logit_margin_lam0p02_top0p50 score_tail_logit_margin_lam0p05_top0p50 score_tail_logit_margin_lam0p02_top0p25 point_pair_tail_logit_margin_lam0p02_top0p50` |  |
| epochs | PASS | `3` |  |
| selection_delta | PASS | `0.005` |  |
| scout_has_no_sample_caps | PASS | `{'MAX_TRAIN_SAMPLES': '', 'MAX_EVAL_SAMPLES': '', 'MAX_TRAIN_PAIRS': '', 'MAX_DEV_PAIRS': ''}` | Exp14 scout runs should not use sample or pair caps. |
| config_score_logit_margin_lam0p01_alllow | PASS | `thesis_exp/configs/exp14_logit_margin_tail_risk_oc/exp14_score_logit_margin_lam0p01_alllow.yaml` |  |
| score_logit_margin_lam0p01_alllow_exp13_probability_hinge_disabled | PASS | `False` | Exp14 must not reuse Exp13 probability-space q3 hinge. |
| score_logit_margin_lam0p01_alllow_logit_margin_enabled | PASS | `True` |  |
| score_logit_margin_lam0p01_alllow_logit_margin_threshold_t | PASS | `3` | threshold tensor index is 2 in the loss implementation |
| score_logit_margin_lam0p01_alllow_selection_rule | PASS | `mae_guard_low_to_high_then_p_gt_3` |  |
| score_logit_margin_lam0p01_alllow_selection_decode_mode | PASS | `projected` |  |
| config_score_logit_margin_lam0p02_alllow | PASS | `thesis_exp/configs/exp14_logit_margin_tail_risk_oc/exp14_score_logit_margin_lam0p02_alllow.yaml` |  |
| score_logit_margin_lam0p02_alllow_exp13_probability_hinge_disabled | PASS | `False` | Exp14 must not reuse Exp13 probability-space q3 hinge. |
| score_logit_margin_lam0p02_alllow_logit_margin_enabled | PASS | `True` |  |
| score_logit_margin_lam0p02_alllow_logit_margin_threshold_t | PASS | `3` | threshold tensor index is 2 in the loss implementation |
| score_logit_margin_lam0p02_alllow_selection_rule | PASS | `mae_guard_low_to_high_then_p_gt_3` |  |
| score_logit_margin_lam0p02_alllow_selection_decode_mode | PASS | `projected` |  |
| config_score_logit_margin_lam0p05_alllow | PASS | `thesis_exp/configs/exp14_logit_margin_tail_risk_oc/exp14_score_logit_margin_lam0p05_alllow.yaml` |  |
| score_logit_margin_lam0p05_alllow_exp13_probability_hinge_disabled | PASS | `False` | Exp14 must not reuse Exp13 probability-space q3 hinge. |
| score_logit_margin_lam0p05_alllow_logit_margin_enabled | PASS | `True` |  |
| score_logit_margin_lam0p05_alllow_logit_margin_threshold_t | PASS | `3` | threshold tensor index is 2 in the loss implementation |
| score_logit_margin_lam0p05_alllow_selection_rule | PASS | `mae_guard_low_to_high_then_p_gt_3` |  |
| score_logit_margin_lam0p05_alllow_selection_decode_mode | PASS | `projected` |  |
| config_score_tail_logit_margin_lam0p02_top0p50 | PASS | `thesis_exp/configs/exp14_logit_margin_tail_risk_oc/exp14_score_tail_logit_margin_lam0p02_top0p50.yaml` |  |
| score_tail_logit_margin_lam0p02_top0p50_exp13_probability_hinge_disabled | PASS | `False` | Exp14 must not reuse Exp13 probability-space q3 hinge. |
| score_tail_logit_margin_lam0p02_top0p50_logit_margin_enabled | PASS | `True` |  |
| score_tail_logit_margin_lam0p02_top0p50_logit_margin_threshold_t | PASS | `3` | threshold tensor index is 2 in the loss implementation |
| score_tail_logit_margin_lam0p02_top0p50_selection_rule | PASS | `mae_guard_low_to_high_then_p_gt_3` |  |
| score_tail_logit_margin_lam0p02_top0p50_selection_decode_mode | PASS | `projected` |  |
| config_score_tail_logit_margin_lam0p05_top0p50 | PASS | `thesis_exp/configs/exp14_logit_margin_tail_risk_oc/exp14_score_tail_logit_margin_lam0p05_top0p50.yaml` |  |
| score_tail_logit_margin_lam0p05_top0p50_exp13_probability_hinge_disabled | PASS | `False` | Exp14 must not reuse Exp13 probability-space q3 hinge. |
| score_tail_logit_margin_lam0p05_top0p50_logit_margin_enabled | PASS | `True` |  |
| score_tail_logit_margin_lam0p05_top0p50_logit_margin_threshold_t | PASS | `3` | threshold tensor index is 2 in the loss implementation |
| score_tail_logit_margin_lam0p05_top0p50_selection_rule | PASS | `mae_guard_low_to_high_then_p_gt_3` |  |
| score_tail_logit_margin_lam0p05_top0p50_selection_decode_mode | PASS | `projected` |  |
| config_score_tail_logit_margin_lam0p02_top0p25 | PASS | `thesis_exp/configs/exp14_logit_margin_tail_risk_oc/exp14_score_tail_logit_margin_lam0p02_top0p25.yaml` |  |
| score_tail_logit_margin_lam0p02_top0p25_exp13_probability_hinge_disabled | PASS | `False` | Exp14 must not reuse Exp13 probability-space q3 hinge. |
| score_tail_logit_margin_lam0p02_top0p25_logit_margin_enabled | PASS | `True` |  |
| score_tail_logit_margin_lam0p02_top0p25_logit_margin_threshold_t | PASS | `3` | threshold tensor index is 2 in the loss implementation |
| score_tail_logit_margin_lam0p02_top0p25_selection_rule | PASS | `mae_guard_low_to_high_then_p_gt_3` |  |
| score_tail_logit_margin_lam0p02_top0p25_selection_decode_mode | PASS | `projected` |  |
| config_point_pair_tail_logit_margin_lam0p02_top0p50 | PASS | `thesis_exp/configs/exp14_logit_margin_tail_risk_oc/exp14_point_pair_tail_logit_margin_lam0p02_top0p50.yaml` |  |
| point_pair_tail_logit_margin_lam0p02_top0p50_exp13_probability_hinge_disabled | PASS | `False` | Exp14 must not reuse Exp13 probability-space q3 hinge. |
| point_pair_tail_logit_margin_lam0p02_top0p50_logit_margin_enabled | PASS | `True` |  |
| point_pair_tail_logit_margin_lam0p02_top0p50_logit_margin_threshold_t | PASS | `3` | threshold tensor index is 2 in the loss implementation |
| point_pair_tail_logit_margin_lam0p02_top0p50_selection_rule | PASS | `mae_guard_low_to_high_then_p_gt_3` |  |
| point_pair_tail_logit_margin_lam0p02_top0p50_selection_decode_mode | PASS | `projected` |  |
