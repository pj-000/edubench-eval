# Schema Profile

Each JSON/JSONL file was inspected using the first five records and a deterministic random sample of
up to twenty records.

## Role-Level Summary

| likely_role | num_files | total_records | example_files |
| --- | --- | --- | --- |
| unknown | 178 | 65630 | ["5-grades/1-shot_cases_zh.json", "5-grades/5_50_metric_v3_questions_... |
| llm_judge_output | 49 | 29931 | ["5-grades/5_groupby_metric_v3_eval_en.jsonl", "5-grades/5_groupby_me... |
| human_annotation | 6 | 20196 | ["5-grades/5_human_1.jsonl", "5-grades/5_human_2.jsonl", "5-grades/5_... |
| merged_human_metric | 6 | 25910 | ["5-grades/5_merge_human_metric_en.jsonl", "5-grades/5_merge_human_me... |
| official_raw | 6 | 4380 | ["download_raw/deepseek-r1_merged.jsonl", "download_raw/deepseek-v3_m... |
| synthetic_or_augmented | 12 | 22204 | ["download_raw/deepseek-r1_pointwise_filtered_en_data_sampled.jsonl",... |
| script_or_config | 2 | 18 | ["edu-data-synthesis-main/data/criteria/metrics_map.json", "metrics_m... |

## Candidate Role Decisions

| file_path | likely_role | decision_note |
| --- | --- | --- |
| 5-grades/1-shot_cases_zh.json | unknown | profiled only |
| 5-grades/5_50_metric_v3_questions_en.json | unknown | profiled only |
| 5-grades/5_50_metric_v3_questions_zh.json | unknown | profiled only |
| 5-grades/5_50_metric_v3_questions_zh_test.json | unknown | profiled only |
| 5-grades/5_groupby_metric_v3_eval_en.jsonl | llm_judge_output | automatic judge output; not a human-label source |
| 5-grades/5_groupby_metric_v3_eval_zh.jsonl | llm_judge_output | automatic judge output; not a human-label source |
| 5-grades/5_human_1.jsonl | human_annotation | real human annotation candidate; useful for corroboration but may use... |
| 5-grades/5_human_2.jsonl | human_annotation | real human annotation candidate; useful for corroboration but may use... |
| 5-grades/5_human_3.jsonl | human_annotation | real human annotation candidate; useful for corroboration but may use... |
| 5-grades/5_merge_human_metric_en.jsonl | merged_human_metric | merged human annotation candidate; profile before using because row m... |
| 5-grades/5_merge_human_metric_zh.jsonl | merged_human_metric | merged human annotation candidate; profile before using because row m... |
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
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_EduBe... | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_deeps... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_deeps... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_gpt-4... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_qwq-p... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_EduBe... | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_deeps... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_deeps... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_gpt-4... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_qwq-p... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_EduBenchEv... | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_deepseek-r... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_deepseek-v... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_gpt-4o.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_2.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_mean... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_qwq-plus.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_EduBenchE... | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_deepseek-... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_deepseek-... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_gpt-4o.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_2.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_mea... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_qwq-plus.... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_EduBenchE... | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_deepseek-... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_deepseek-... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_gpt-4o.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_2.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_mea... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_qwq-plus.... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_EduBenchE... | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_deepseek-... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_deepseek-... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_gpt-4o.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_human_1.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_human_2.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_human_3.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_human_mea... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_qwq-plus.... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_EduBen... | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_deepse... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_deepse... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_gpt-4o... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_human_... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_human_... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_human_... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_human_... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_qwq-pl... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_EduBench... | llm_judge_output | automatic judge output; not a human-label source |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_deepseek... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_deepseek... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_gpt-4o.json | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_human_1.... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_human_2.... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_human_3.... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_human_me... | unknown | profiled only |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_qwq-plus... | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_EduBenc... | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_deepsee... | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_deepsee... | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_gpt-4o.... | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_1... | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_2... | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_3... | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_m... | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_qwq-plu... | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_EduBenchEvalu... | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_deepseek-r1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_deepseek-v3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_gpt-4o.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_human_1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_human_2.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_human_3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_human_mean.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_qwq-plus.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_EduBenchEvalu... | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_deepseek-r1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_deepseek-v3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_gpt-4o.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_human_1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_human_2.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_human_3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_human_mean.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_qwq-plus.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_EduBenchEvaluator.... | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_deepseek-r1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_deepseek-v3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_gpt-4o.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_human_1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_human_2.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_human_3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_human_mean.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_qwq-plus.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_1_EduBenchEvaluator... | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/human_1_deepseek-r1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_1_deepseek-v3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_1_gpt-4o.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_1_human_1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_1_human_2.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_1_human_3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_1_human_mean.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_1_qwq-plus.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_2_EduBenchEvaluator... | llm_judge_output | automatic judge output; not a human-label source |
| correlation/corr_res_kendall_split_and_fill/human_2_deepseek-r1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_2_deepseek-v3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_2_gpt-4o.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_2_human_1.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_2_human_2.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_2_human_3.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_2_human_mean.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_2_qwq-plus.json | unknown | profiled only |
| correlation/corr_res_kendall_split_and_fill/human_3_EduBenchEvaluator... | llm_judge_output | automatic judge output; not a human-label source |

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
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_EduBe... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_deeps... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_deeps... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_gpt-4... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_qwq-p... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_EduBe... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_deeps... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_deeps... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_gpt-4... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_qwq-p... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_EduBenchEv... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_deepseek-r... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_deepseek-v... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_mean... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_EduBenchE... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_deepseek-... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_deepseek-... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_mea... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_qwq-plus.... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_EduBenchE... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_deepseek-... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_deepseek-... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_mea... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_qwq-plus.... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_EduBenchE... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_deepseek-... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_deepseek-... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_human_mea... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_qwq-plus.... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_EduBen... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_deepse... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_deepse... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_gpt-4o... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_human_... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_human_... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_human_... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_human_... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/human_mean_qwq-pl... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_EduBench... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_deepseek... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_deepseek... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_human_1.... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_human_2.... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_human_3.... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_human_me... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation-testset/corr_res_kendall_split_and_fill/qwq-plus_qwq-plus... | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_EduBenc... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_deepsee... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_deepsee... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_gpt-4o.... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_1... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_2... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_3... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_m... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/EduBenchEvaluator_qwq-plu... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_EduBenchEvalu... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-r1_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_EduBenchEvalu... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/deepseek-v3_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_EduBenchEvaluator.... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/gpt-4o_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_EduBenchEvaluator... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_1_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_EduBenchEvaluator... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_deepseek-r1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_deepseek-v3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_gpt-4o.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_human_1.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_human_2.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_human_3.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_human_mean.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_2_qwq-plus.json | 14 | unknown | [] | [] | [] | [] | [] | [] |
| correlation/corr_res_kendall_split_and_fill/human_3_EduBenchEvaluator... | 14 | llm_judge_output | [] | [] | [] | [] | [] | [] |

_Showing 160 of 259 rows._
