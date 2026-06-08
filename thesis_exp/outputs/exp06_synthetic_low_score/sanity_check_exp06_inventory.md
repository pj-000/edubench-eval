# Exp6 Inventory Sanity Check

| check | status | observed | expected | notes |
| --- | --- | --- | --- | --- |
| synthetic_source_inventory.csv exists | PASS | True | True |  |
| synthetic_source_inventory.csv required columns | PASS | [] | [] |  |
| synthetic_schema_profile.csv exists | PASS | True | True |  |
| synthetic_schema_profile.csv required columns | PASS | [] | [] |  |
| synthetic_candidate_rows.csv exists | PASS | True | True |  |
| synthetic_candidate_rows.csv required columns | PASS | [] | [] |  |
| synthetic_leakage_summary.csv exists | PASS | True | True |  |
| synthetic_leakage_summary.csv required columns | PASS | [] | [] |  |
| synthetic_leakage_details.csv exists | PASS | True | True |  |
| synthetic_leakage_details.csv required columns | PASS | [] | [] |  |
| synthetic_score_distribution.csv exists | PASS | True | True |  |
| synthetic_score_distribution.csv required columns | PASS | [] | [] |  |
| synthetic_metric_distribution.csv exists | PASS | True | True |  |
| synthetic_metric_distribution.csv required columns | PASS | [] | [] |  |
| synthetic_language_distribution.csv exists | PASS | True | True |  |
| synthetic_language_distribution.csv required columns | PASS | [] | [] |  |
| synthetic_error_type_distribution.csv exists | PASS | True | True |  |
| synthetic_error_type_distribution.csv required columns | PASS | [] | [] |  |
| synthetic_filter_recommendation.csv exists | PASS | True | True |  |
| synthetic_filter_recommendation.csv required columns | PASS | [] | [] |  |
| report.md exists | PASS | True | True |  |
| review_package.md exists | PASS | True | True |  |
| notion_exp06_inventory_summary.md exists | PASS | True | True |  |
| sampled_merge_50_new.json default HIGH risk | PASS | HIGH | HIGH |  |
| sampled_merge_50_new_swift.json default HIGH risk | PASS | HIGH | HIGH |  |
| merge_model_metric.jsonl blocked judge role | PASS | ["model_judge_output", "NO_JUDGE_OUTPUT_ONLY"] | model_judge_output / NO* |  |
| deepseek-r1_merged.jsonl blocked judge role | PASS | ["model_judge_output", "NO_JUDGE_OUTPUT_ONLY"] | model_judge_output / NO* |  |
| groupby_metric_qwq_eval_en.jsonl blocked judge role | PASS | ["model_judge_output", "NO_JUDGE_OUTPUT_ONLY"] | model_judge_output / NO* |  |
| groupby_metric_qwq_eval_zh.jsonl blocked judge role | PASS | ["model_judge_output", "NO_JUDGE_OUTPUT_ONLY"] | model_judge_output / NO* |  |
| groupby_metric_r1_eval_en.jsonl blocked judge role | PASS | ["model_judge_output", "NO_JUDGE_OUTPUT_ONLY"] | model_judge_output / NO* |  |
| groupby_metric_r1_eval_zh.jsonl blocked judge role | PASS | ["model_judge_output", "NO_JUDGE_OUTPUT_ONLY"] | model_judge_output / NO* |  |
| groupby_metric_v3_eval_en.jsonl blocked judge role | PASS | ["model_judge_output", "NO_JUDGE_OUTPUT_ONLY"] | model_judge_output / NO* |  |
| groupby_metric_v3_eval_zh.jsonl blocked judge role | PASS | ["model_judge_output", "NO_JUDGE_OUTPUT_ONLY"] | model_judge_output / NO* |  |
| candidate rows generated | PASS | 106457 | >0 |  |
| target_label_5 rows profiled | PASS | 75109 | >0 | warning allows no-label repositories but Exp6 needs labels |
| low-score rows profiled | PASS | 6687 | >0 | Exp6 low-score augmentation requires low labels |
| no Exp6 synthetic train/dev/test generated | PASS | [] | [] |  |
| Exp0 split train row count unchanged | PASS | 2654 | 2654 |  |
| Exp0 split dev row count unchanged | PASS | 664 | 664 |  |
| Exp0 split test row count unchanged | PASS | 2218 | 2218 |  |
