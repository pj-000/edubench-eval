# Exp5 Setup Sanity Check

Overall status: **PASS**

| check | status | observed | expected | notes |
| --- | --- | --- | --- | --- |
| Exp4 O3 baseline outputs exist | PASS | [] | [] |  |
| A4 dataset exists | PASS | thesis_exp/outputs/exp04_target_objectives/datasets/A4_fixed_question_answer_metric_rubric_metadata | Exp4 fixed A4 |  |
| train rows | PASS | 2654 | 2654 |  |
| dev rows | PASS | 664 | 664 |  |
| test rows | PASS | 2218 | 2218 |  |
| train label_5 values | PASS | [1, 2, 3, 4, 5] | [1, 2, 3, 4, 5] |  |
| train no synthetic/sample data path | PASS | thesis_exp/outputs/exp04_target_objectives/datasets/A4_fixed_question_answer_metric_rubric_metadata | fixed A4 path |  |
| dev label_5 values | PASS | [1, 2, 3, 4, 5] | [1, 2, 3, 4, 5] |  |
| dev no synthetic/sample data path | PASS | thesis_exp/outputs/exp04_target_objectives/datasets/A4_fixed_question_answer_metric_rubric_metadata | fixed A4 path |  |
| test label_5 values | PASS | [1, 2, 3, 4, 5] | [1, 2, 3, 4, 5] |  |
| test no synthetic/sample data path | PASS | thesis_exp/outputs/exp04_target_objectives/datasets/A4_fixed_question_answer_metric_rubric_metadata | fixed A4 path |  |
| class_weights.csv exists | PASS | thesis_exp/outputs/exp05_low_score_loss/tables/class_weights.csv | exists |  |
| w_min/w_max | PASS | 0.5/3.0 | 0.5/3.0 |  |
| class weights use train only | PASS | thesis_exp/outputs/exp04_target_objectives/datasets/A4_fixed_question_answer_metric_rubric_metadata/train.j... | train split |  |
| all class weights finite | PASS | [(1, 3.0), (2, 3.0), (3, 2.1147410358565737), (4, 0.5610993657505285), (5, 0.5)] | finite labels 1..5 |  |
| toy ordinal targets | PASS | [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]] | [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]] |  |
| toy weighted ordinal loss finite | PASS | 0.6931471824645996 | finite non-NaN | {'mean_base_loss': 0.6931471824645996, 'mean_weighted_loss': 1.0050634145736694, 'mean_sample_weight': 1.45... |
| L2 config exists exp05_l2a_asymmetric_ordinal_lambda03.yaml | PASS | thesis_exp/configs/exp05_low_score_loss/exp05_l2a_asymmetric_ordinal_lambda03.yaml | exists |  |
| L2 config exists exp05_l2b_asymmetric_ordinal_lambda05.yaml | PASS | thesis_exp/configs/exp05_low_score_loss/exp05_l2b_asymmetric_ordinal_lambda05.yaml | exists |  |
| L2 config exists exp05_l2_smoke_test.yaml | PASS | thesis_exp/configs/exp05_low_score_loss/exp05_l2_smoke_test.yaml | exists |  |
| exp05_l2a_asymmetric_ordinal_lambda03.yaml lambda_low > 0 | PASS | 0.3 | >0 |  |
| exp05_l2a_asymmetric_ordinal_lambda03.yaml margin >= 0 | PASS | 0.0 | >=0 |  |
| exp05_l2a_asymmetric_ordinal_lambda03.yaml use_class_weights=false | PASS | false | false |  |
| exp05_l2a_asymmetric_ordinal_lambda03.yaml use_high_score_preservation=false | PASS | false | false |  |
| exp05_l2a_asymmetric_ordinal_lambda03.yaml use_threshold_suppression=false | PASS | false | false |  |
| exp05_l2b_asymmetric_ordinal_lambda05.yaml lambda_low > 0 | PASS | 0.5 | >0 |  |
| exp05_l2b_asymmetric_ordinal_lambda05.yaml margin >= 0 | PASS | 0.0 | >=0 |  |
| exp05_l2b_asymmetric_ordinal_lambda05.yaml use_class_weights=false | PASS | false | false |  |
| exp05_l2b_asymmetric_ordinal_lambda05.yaml use_high_score_preservation=false | PASS | false | false |  |
| exp05_l2b_asymmetric_ordinal_lambda05.yaml use_threshold_suppression=false | PASS | false | false |  |
| L2 toy loss checks | PASS | thesis_exp/outputs/exp05_low_score_loss/tables/l2_toy_loss_checks.csv | all PASS |  |
| L2 penalty only applies to label_5 <= 2 | PASS | [0.809999942779541, 0.05062501132488251, 0.0, 0.0, 0.0] | non-low penalties zero |  |
| L2 does not modify class_weights.csv | PASS | thesis_exp/outputs/exp05_low_score_loss/tables/class_weights.csv | existing L1 weights retained |  |
| L1 outputs remain present | PASS | thesis_exp/outputs/exp05_low_score_loss/runs/L1_weighted_ordinal/tables/metrics_summary.csv | L1_weighted_ordinal metrics |  |
| bash -n thesis_exp/scripts/run_exp05_l1_smoke.sh | PASS | ok | ok |  |
| bash -n thesis_exp/scripts/run_exp05_l1_train.sh | PASS | ok | ok |  |
| bash -n thesis_exp/scripts/run_exp05_l2_smoke.sh | PASS | ok | ok |  |
| bash -n thesis_exp/scripts/run_exp05_l2_train.sh | PASS | ok | ok |  |
| exp05 Python modules py_compile | PASS | ok | ok |  |
| checkpoint artifacts path gitignored | PASS | [] | all required patterns |  |
| Exp5 artifacts directory | PASS | thesis_exp/artifacts/exp05_low_score_loss | under thesis_exp/artifacts |  |
| no checkpoint/weights tracked by git | PASS | [] | [] |  |
