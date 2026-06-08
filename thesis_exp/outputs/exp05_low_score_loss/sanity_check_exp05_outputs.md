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
| L2a_asymmetric_ordinal_lambda03_margin0 run status | PASS | pending | pending allowed |  |
| L2b_asymmetric_ordinal_lambda05_margin0 run status | PASS | pending | pending allowed |  |
| L0/L1 metrics present in summary | PASS | {'L0_exp04_o3_ordinal': 'completed', 'L1_weighted_ordinal': 'completed', 'L2a_asymmetric_ordinal_... | L0 and L1 present |  |
| no checkpoint/weights tracked | PASS | [] | [] |  |
