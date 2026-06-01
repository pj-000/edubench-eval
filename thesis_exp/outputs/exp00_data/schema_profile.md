# Schema Profile

Each JSON/JSONL file was inspected using the first five records and a deterministic random sample of up to twenty records.

## Role-Level Summary

| likely_role | num_files | total_records | example_files |
| --- | --- | --- | --- |
| unknown | 178 | 65630 | ["5-grades/1-shot_cases_zh.json", "5-grades/5_50_metric_v3_questions_en.json", "5-grades/5_50_metric_v3_questions_zh.json", "5-grades/5_50_metric_v3_questions_zh_test.json", "5-... |
| llm_judge_output | 49 | 29931 | ["5-grades/5_groupby_metric_v3_eval_en.jsonl", "5-grades/5_groupby_metric_v3_eval_zh.jsonl", "5-grades/judge_v3_questions_zh.jsonl", "correlation-testset/corr_res_kendall_split_... |
| human_annotation | 6 | 20196 | ["5-grades/5_human_1.jsonl", "5-grades/5_human_2.jsonl", "5-grades/5_human_3.jsonl", "human_1.jsonl", "human_2.jsonl"] |
| merged_human_metric | 6 | 25910 | ["5-grades/5_merge_human_metric_en.jsonl", "5-grades/5_merge_human_metric_zh.jsonl", "merge_human_metric.jsonl", "merge_human_metric_strict_en.jsonl", "merge_human_metric_strict... |
| official_raw | 6 | 4380 | ["download_raw/deepseek-r1_merged.jsonl", "download_raw/deepseek-v3_merged.jsonl", "download_raw/gpt-4o_merged.jsonl", "download_raw/grouped_by_metric.jsonl", "download_raw/grou... |
| synthetic_or_augmented | 12 | 22204 | ["download_raw/deepseek-r1_pointwise_filtered_en_data_sampled.jsonl", "download_raw/deepseek-r1_pointwise_filtered_zh_data_sampled.jsonl", "download_raw/deepseek-v3_pointwise_fi... |
| script_or_config | 2 | 18 | ["edu-data-synthesis-main/data/criteria/metrics_map.json", "metrics_map.json"] |

## Candidate Role Decisions

| file_path | likely_role | decision_note |
| --- | --- | --- |
| 5-grades/1-shot_cases_zh.json | unknown | profiled only |
| 5-grades/5_50_metric_v3_questions_en.json | unknown | profiled only |
| 5-grades/5_50_metric_v3_questions_zh.json | unknown | profiled only |
| 5-grades/5_50_metric_v3_questions_zh_test.json | unknown | profiled only |
| 5-grades/5_groupby_metric_v3_eval_en.jsonl | llm_judge_output | automatic judge output; not a human-label source |
| 5-grades/5_groupby_metric_v3_eval_zh.jsonl | llm_judge_output | automatic judge output; not a human-label source |
| 5-grades/5_human_1.jsonl | human_annotation | real human annotation candidate; useful for corroboration but may use a different 10-point scale |
| 5-grades/5_human_2.jsonl | human_annotation | real human annotation candidate; useful for corroboration but may use a different 10-point scale |
| 5-grades/5_human_3.jsonl | human_annotation | real human annotation candidate; useful for corroboration but may use a different 10-point scale |
| 5-grades/5_merge_human_metric_en.jsonl | merged_human_metric | merged human annotation candidate; profile before using because row multiplicity may differ |
| 5-grades/5_merge_human_metric_zh.jsonl | merged_human_metric | merged human annotation candidate; profile before using because row multiplicity may differ |
| 5-grades/5_metrics_en.json | unknown | profiled only |
| 5-grades/5_metrics_zh.json | unknown | profiled only |
| 5-grades/Untitled-1.json | unknown | profiled only |
| 5-grades/example.jsonl | unknown | profiled only |
| 5-grades/example_zh.jsonl | unknown | profiled only |
| 5-grades/judge_v3_questions_zh.jsonl | llm_judge_output | automatic judge output; not a human-label source |
| analysis_outputs/manifest.json | unknown | profiled only |
| categories/category.json | unknown | profiled only |
| categories/category_merge.json | unknown | profiled only |
| categories/category_merge_1.json | unknown | profiled only |
| categories/category_merge_2.json | unknown | profiled only |
| categories/category_no_design.json | unknown | profiled only |
| categories/category_reorganized.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_EduBenchEvaluator.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_deepseek-r1.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_deepseek-v3.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_gpt-4o.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_1.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_2.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_3.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_mean.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_qwq-plus.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_EduBenchEvaluator.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_deepseek-r1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_deepseek-v3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_gpt-4o.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human_1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human_2.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human_3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human_mean.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_qwq-plus.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_EduBenchEvaluator.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_deepseek-r1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_deepseek-v3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_gpt-4o.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human_1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human_2.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human_3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human_mean.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_qwq-plus.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_EduBenchEvaluator.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_deepseek-r1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_deepseek-v3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_gpt-4o.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_2.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_mean.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_qwq-plus.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_EduBenchEvaluator.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_deepseek-r1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_deepseek-v3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_gpt-4o.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_2.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_mean.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_qwq-plus.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_EduBenchEvaluator.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_deepseek-r1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_deepseek-v3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_gpt-4o.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_2.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_mean.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_qwq-plus.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_EduBenchEvaluator.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_deepseek-r1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_deepseek-v3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_gpt-4o.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_human_1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_human_2.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_human_3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_human_mean.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_qwq-plus.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_EduBenchEvaluator.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_deepseek-r1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_deepseek-v3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_gpt-4o.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_human_1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_human_2.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_human_3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_human_mean.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_qwq-plus.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_EduBenchEvaluator.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_deepseek-r1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_deepseek-v3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_gpt-4o.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_human_1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_human_2.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_human_3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_human_mean.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_qwq-plus.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_EduBenchEvaluator.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_deepseek-r1.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_deepseek-v3.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_gpt-4o.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_1.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_2.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_3.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_mean.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_qwq-plus.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_EduBenchEvaluator.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_deepseek-r1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_deepseek-v3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_gpt-4o.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_human_1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_human_2.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_human_3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_human_mean.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_qwq-plus.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_EduBenchEvaluator.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_deepseek-r1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_deepseek-v3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_gpt-4o.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_human_1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_human_2.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_human_3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_human_mean.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_qwq-plus.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_EduBenchEvaluator.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_deepseek-r1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_deepseek-v3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_gpt-4o.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_human_1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_human_2.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_human_3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_human_mean.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_qwq-plus.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_1_EduBenchEvaluator.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/human_1_deepseek-r1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_1_deepseek-v3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_1_gpt-4o.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_1_human_1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_1_human_2.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_1_human_3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_1_human_mean.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_1_qwq-plus.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_2_EduBenchEvaluator.json | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/human_2_deepseek-r1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_2_deepseek-v3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_2_gpt-4o.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_2_human_1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_2_human_2.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_2_human_3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_2_human_mean.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_2_qwq-plus.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_3_EduBenchEvaluator.json | llm_judge_output | automatic judge output; not a human-label source |

_Showing 160 of 259 rows._

## Profile Rows

| file_path | num_records | likely_role | score_like_fields | question_like_fields | answer_like_fields | metric_like_fields | scenario_like_fields | human_score_like_fields |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5-grades/1-shot_cases_zh.json | 12 | unknown | [] | ["question"] | ["generated_responses"] | ["principle"] | [] | [] |
| 5-grades/5_50_metric_v3_questions_en.json | 12 | unknown | [] | [] | [] | [] | [] | [] |
| 5-grades/5_50_metric_v3_questions_zh.json | 12 | unknown | [] | [] | [] | [] | [] | [] |
| 5-grades/5_50_metric_v3_questions_zh_test.json | 2 | unknown | [] | [] | [] | [] | [] | [] |
| 5-grades/5_groupby_metric_v3_eval_en.jsonl | 2785 | llm_judge_output | ["score"] | ["question"] | ["response"] | ["principle"] | [] | [] |
| 5-grades/5_groupby_metric_v3_eval_zh.jsonl | 2805 | llm_judge_output | ["score"] | ["question"] | ["response"] | ["principle"] | [] | [] |
| 5-grades/5_human_1.jsonl | 3366 | human_annotation | ["score"] | ["question"] | ["response"] | ["principle"] | [] | [] |
| 5-grades/5_human_2.jsonl | 3366 | human_annotation | ["score"] | ["question"] | ["response"] | ["principle"] | [] | [] |
| 5-grades/5_human_3.jsonl | 3366 | human_annotation | ["score"] | ["question"] | ["response"] | ["principle"] | [] | [] |
| 5-grades/5_merge_human_metric_en.jsonl | 4031 | merged_human_metric | ["score"] | ["question"] | ["response"] | ["principle"] | [] | [] |
| 5-grades/5_merge_human_metric_zh.jsonl | 4007 | merged_human_metric | ["score"] | ["question"] | ["response"] | ["principle"] | [] | [] |
| 5-grades/5_metrics_en.json | 12 | unknown | [] | [] | [] | [] | [] | [] |
| 5-grades/5_metrics_zh.json | 12 | unknown | [] | [] | [] | [] | [] | [] |
| 5-grades/Untitled-1.json | 3 | unknown | ["score", "scores"] | ["question"] | ["response"] | ["principle"] | ["task"] | [] |
| 5-grades/example.jsonl | 0 | unknown | [] | [] | [] | [] | [] | [] |
| 5-grades/example_zh.jsonl | 75 | unknown | ["score"] | ["question"] | ["response"] | ["principle"] | [] | [] |
| 5-grades/judge_v3_questions_zh.jsonl | 0 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| analysis_outputs/manifest.json | 4 | unknown | [] | [] | [] | [] | [] | [] |
| categories/category.json | 18 | unknown | [] | [] | [] | [] | [] | [] |
| categories/category_merge.json | 16 | unknown | [] | [] | [] | [] | [] | [] |
| categories/category_merge_1.json | 4 | unknown | [] | [] | [] | [] | [] | [] |
| categories/category_merge_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| categories/category_no_design.json | 16 | unknown | [] | [] | [] | [] | [] | [] |
| categories/category_reorganized.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_EduBenchEvaluator.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_deepseek-r1.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_deepseek-v3.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_gpt-4o.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_1.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_2.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_3.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_mean.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_qwq-plus.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_EduBenchEvaluator.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_EduBenchEvaluator.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_EduBenchEvaluator.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_EduBenchEvaluator.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_EduBenchEvaluator.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_EduBenchEvaluator.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_EduBenchEvaluator.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_EduBenchEvaluator.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_EduBenchEvaluator.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_deepseek-r1.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_deepseek-v3.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_gpt-4o.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_1.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_2.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_3.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_mean.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_qwq-plus.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_EduBenchEvaluator.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_EduBenchEvaluator.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_EduBenchEvaluator.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_EduBenchEvaluator.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_EduBenchEvaluator.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_3_EduBenchEvaluator.json | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |

_Showing 160 of 259 rows._
