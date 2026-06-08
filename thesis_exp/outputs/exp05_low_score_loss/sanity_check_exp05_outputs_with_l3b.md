# Exp5 Output Sanity Check

Overall status: **PASS**

| check | status | observed | expected | notes |
| --- | --- | --- | --- | --- |
| L1_weighted_ordinal run status | PASS | completed | completed |  |
| L1_weighted_ordinal predictions_dev | PASS | rows=664 missing=[] | readable JSONL |  |
| L1_weighted_ordinal predictions_test | PASS | rows=2218 missing=[] | readable JSONL |  |
| L1_weighted_ordinal dev_test_arrays.npz | PASS | test_shape=(2218, 4) missing=[] | required array keys, logits dim=4 |  |
| L1_weighted_ordinal metrics_summary.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L1_weighted_ordinal/tables/metrics_summary.csv | exists |  |
| L1_weighted_ordinal per_bin_metrics.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L1_weighted_ordinal/tables/per_bin_metrics.csv | exists |  |
| L1_weighted_ordinal low_score_metrics.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L1_weighted_ordinal/tables/low_score_metrics.csv | exists |  |
| L1_weighted_ordinal high_score_metrics.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L1_weighted_ordinal/tables/high_score_metrics.csv | exists |  |
| L1_weighted_ordinal metric_level_metrics.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L1_weighted_ordinal/tables/metric_level_metrics.csv | exists |  |
| L1_weighted_ordinal scenario_level_metrics.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L1_weighted_ordinal/tables/scenario_level_metrics.csv | exists |  |
| L1_weighted_ordinal score_distribution.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L1_weighted_ordinal/tables/score_distribution.csv | exists |  |
| L1_weighted_ordinal selection metric | PASS | dev_MAE_label | dev_MAE_label |  |
| L1_weighted_ordinal checkpoint selection | PASS | epoch=9 value=0.410140562248996 | dev_MAE_label min=0.410140562248996 |  |
| L2a_asymmetric_ordinal_lambda03_margin0 run status | PASS | completed | completed |  |
| L2a_asymmetric_ordinal_lambda03_margin0 predictions_dev | PASS | rows=664 missing=[] | readable JSONL |  |
| L2a_asymmetric_ordinal_lambda03_margin0 predictions_test | PASS | rows=2218 missing=[] | readable JSONL |  |
| L2a_asymmetric_ordinal_lambda03_margin0 dev_test_arrays.npz | PASS | test_shape=(2218, 4) missing=[] | required array keys, logits dim=4 |  |
| L2a_asymmetric_ordinal_lambda03_margin0 metrics_summary.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2a_asymmetric_ordinal_lambda03_margin0/tables/metri... | exists |  |
| L2a_asymmetric_ordinal_lambda03_margin0 per_bin_metrics.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2a_asymmetric_ordinal_lambda03_margin0/tables/per_b... | exists |  |
| L2a_asymmetric_ordinal_lambda03_margin0 low_score_metrics.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2a_asymmetric_ordinal_lambda03_margin0/tables/low_s... | exists |  |
| L2a_asymmetric_ordinal_lambda03_margin0 high_score_metrics.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2a_asymmetric_ordinal_lambda03_margin0/tables/high_... | exists |  |
| L2a_asymmetric_ordinal_lambda03_margin0 metric_level_metrics.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2a_asymmetric_ordinal_lambda03_margin0/tables/metri... | exists |  |
| L2a_asymmetric_ordinal_lambda03_margin0 scenario_level_metrics.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2a_asymmetric_ordinal_lambda03_margin0/tables/scena... | exists |  |
| L2a_asymmetric_ordinal_lambda03_margin0 score_distribution.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2a_asymmetric_ordinal_lambda03_margin0/tables/score... | exists |  |
| L2a_asymmetric_ordinal_lambda03_margin0 loss_debug_history.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2a_asymmetric_ordinal_lambda03_margin0/logs/loss_de... | exists |  |
| L2a_asymmetric_ordinal_lambda03_margin0 loss_debug columns | PASS | epoch,step,loss,mean_base_loss,mean_penalty,lambda_low,margin,active_low_count,active_penalty_cou... | ['mean_base_loss', 'mean_penalty', 'mean_s_hat_low', 'mean_over_low'] |  |
| L2a_asymmetric_ordinal_lambda03_margin0 config recorded in run_summary | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2a_asymmetric_ordinal_lambda03_margin0/run_summary.md | ['lambda_low', 'margin', 'use_class_weights: `false`'] |  |
| L2a_asymmetric_ordinal_lambda03_margin0 selection metric | PASS | dev_MAE_label | dev_MAE_label |  |
| L2a_asymmetric_ordinal_lambda03_margin0 test after best checkpoint | PASS | True | true |  |
| L2a_asymmetric_ordinal_lambda03_margin0 checkpoint selection | PASS | epoch=8 value=0.3805220883534136 | dev_MAE_label min=0.3805220883534136 |  |
| L2b_asymmetric_ordinal_lambda05_margin0 run status | PASS | completed | completed |  |
| L2b_asymmetric_ordinal_lambda05_margin0 predictions_dev | PASS | rows=664 missing=[] | readable JSONL |  |
| L2b_asymmetric_ordinal_lambda05_margin0 predictions_test | PASS | rows=2218 missing=[] | readable JSONL |  |
| L2b_asymmetric_ordinal_lambda05_margin0 dev_test_arrays.npz | PASS | test_shape=(2218, 4) missing=[] | required array keys, logits dim=4 |  |
| L2b_asymmetric_ordinal_lambda05_margin0 metrics_summary.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2b_asymmetric_ordinal_lambda05_margin0/tables/metri... | exists |  |
| L2b_asymmetric_ordinal_lambda05_margin0 per_bin_metrics.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2b_asymmetric_ordinal_lambda05_margin0/tables/per_b... | exists |  |
| L2b_asymmetric_ordinal_lambda05_margin0 low_score_metrics.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2b_asymmetric_ordinal_lambda05_margin0/tables/low_s... | exists |  |
| L2b_asymmetric_ordinal_lambda05_margin0 high_score_metrics.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2b_asymmetric_ordinal_lambda05_margin0/tables/high_... | exists |  |
| L2b_asymmetric_ordinal_lambda05_margin0 metric_level_metrics.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2b_asymmetric_ordinal_lambda05_margin0/tables/metri... | exists |  |
| L2b_asymmetric_ordinal_lambda05_margin0 scenario_level_metrics.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2b_asymmetric_ordinal_lambda05_margin0/tables/scena... | exists |  |
| L2b_asymmetric_ordinal_lambda05_margin0 score_distribution.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2b_asymmetric_ordinal_lambda05_margin0/tables/score... | exists |  |
| L2b_asymmetric_ordinal_lambda05_margin0 loss_debug_history.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2b_asymmetric_ordinal_lambda05_margin0/logs/loss_de... | exists |  |
| L2b_asymmetric_ordinal_lambda05_margin0 loss_debug columns | PASS | epoch,step,loss,mean_base_loss,mean_penalty,lambda_low,margin,active_low_count,active_penalty_cou... | ['mean_base_loss', 'mean_penalty', 'mean_s_hat_low', 'mean_over_low'] |  |
| L2b_asymmetric_ordinal_lambda05_margin0 config recorded in run_summary | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L2b_asymmetric_ordinal_lambda05_margin0/run_summary.md | ['lambda_low', 'margin', 'use_class_weights: `false`'] |  |
| L2b_asymmetric_ordinal_lambda05_margin0 selection metric | PASS | dev_MAE_label | dev_MAE_label |  |
| L2b_asymmetric_ordinal_lambda05_margin0 test after best checkpoint | PASS | True | true |  |
| L2b_asymmetric_ordinal_lambda05_margin0 checkpoint selection | PASS | epoch=10 value=0.38253012048192764 | dev_MAE_label min=0.38253012048192764 |  |
| L3b_weighted_threshold_mu03 run status | PASS | pending | pending allowed |  |
| configured metrics present in summary | PASS | {'L0_exp04_o3_ordinal': 'completed', 'L1_weighted_ordinal': 'completed', 'L2a_asymmetric_ordinal_... | ['L0_exp04_o3_ordinal', 'L1_weighted_ordinal', 'L2a_asymmetric_ordinal_lambda03_margin0', 'L2b_as... |  |
| no checkpoint/weights tracked | PASS | [] | [] |  |
