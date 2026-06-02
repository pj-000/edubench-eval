# Exp2 Train Setup Sanity Check

Overall status: **PASS**

| check | status | observed | expected | notes |
| --- | --- | --- | --- | --- |
| train row count | PASS | 2654 | 2654 |  |
| train label range 0..4 | PASS | [0, 1, 2, 3, 4] | 0..4 |  |
| train label_5 range 1..5 | PASS | [1, 2, 3, 4, 5] | 1..5 |  |
| train template_name | PASS | ['qa_metric_baseline'] | qa_metric_baseline |  |
| train prompt required fields | PASS | 0 | 0 |  |
| train prompt excludes rubric/metadata | PASS | 0 | 0 |  |
| dev row count | PASS | 664 | 664 |  |
| dev label range 0..4 | PASS | [0, 1, 2, 3, 4] | 0..4 |  |
| dev label_5 range 1..5 | PASS | [1, 2, 3, 4, 5] | 1..5 |  |
| dev template_name | PASS | ['qa_metric_baseline'] | qa_metric_baseline |  |
| dev prompt required fields | PASS | 0 | 0 |  |
| dev prompt excludes rubric/metadata | PASS | 0 | 0 |  |
| test row count | PASS | 2218 | 2218 |  |
| test label range 0..4 | PASS | [0, 1, 2, 3, 4] | 0..4 |  |
| test label_5 range 1..5 | PASS | [1, 2, 3, 4, 5] | 1..5 |  |
| test template_name | PASS | ['qa_metric_baseline'] | qa_metric_baseline |  |
| test prompt required fields | PASS | 0 | 0 |  |
| test prompt excludes rubric/metadata | PASS | 0 | 0 |  |
| train/dev record_id overlap | PASS | 0 | 0 |  |
| train/test record_id overlap | PASS | 0 | 0 |  |
| dev/test record_id overlap | PASS | 0 | 0 |  |
| train/dev triple_key overlap | PASS | 0 | 0 |  |
| train/test triple_key overlap | PASS | 0 | 0 |  |
| dev/test triple_key overlap | PASS | 0 | 0 |  |
| .gitignore model artifact coverage | PASS | [] | all required patterns present | checks root .gitignore and thesis_exp/.gitignore |
| bash -n run_exp02_train_ce_0_6b.sh | PASS | ok | ok |  |
| py_compile exp02 modules | PASS | ok | ok |  |
