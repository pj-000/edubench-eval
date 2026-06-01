# Source Inventory

This inventory scans source-like files in the repository while excluding generated `thesis_exp` outputs.

## Role Counts

| likely_role | num_files |
| --- | --- |
| unknown | 294 |
| llm_judge_output | 71 |
| script_or_config | 69 |
| synthetic_or_augmented | 12 |
| official_raw | 8 |
| human_annotation | 6 |
| merged_human_metric | 6 |

## Required Candidate Files

| file_path | file_type | exists | num_rows_or_records | likely_role | sample_keys |
| --- | --- | --- | --- | --- | --- |
| 5-grades.py | py | True |  | script_or_config | [] |
| EduBench.zip | zip | True | 12 | official_raw | [".bib", ".bst", ".md", ".pdf", ".sty", ".tex", ".txt", "<no_ext>"] |
| edu-data-synthesis-main/data/criteria/metrics_map.json | json | True | 9 | script_or_config | ["__key__", "__value__"] |
| groupby_metric_qwq_eval_en.jsonl | jsonl | True | 2709 | llm_judge_output | ["model", "principle", "question", "reason", "response", "score"] |
| groupby_metric_qwq_eval_zh.jsonl | jsonl | True | 2732 | llm_judge_output | ["model", "principle", "question", "reason", "response", "score"] |
| groupby_metric_r1_eval_en.jsonl | jsonl | True | 2767 | llm_judge_output | ["model", "principle", "question", "reason", "response", "score"] |
| groupby_metric_r1_eval_zh.jsonl | jsonl | True | 2624 | llm_judge_output | ["model", "principle", "question", "reason", "response", "score"] |
| groupby_metric_v3_eval_en.jsonl | jsonl | True | 2785 | llm_judge_output | ["model", "principle", "question", "reason", "response", "score"] |
| groupby_metric_v3_eval_zh.jsonl | jsonl | True | 2805 | llm_judge_output | ["model", "principle", "question", "reason", "response", "score"] |
| human_1.jsonl | jsonl | True | 3366 | human_annotation | ["model", "principle", "question", "reason", "response", "score"] |
| human_2.jsonl | jsonl | True | 3366 | human_annotation | ["model", "principle", "question", "reason", "response", "score"] |
| human_3.jsonl | jsonl | True | 3366 | human_annotation | ["model", "principle", "question", "reason", "response", "score"] |
| merge_human_metric.jsonl | jsonl | True | 7549 | merged_human_metric | ["eval", "model", "principle", "question", "reason", "response", "score", "score_mean"] |
| merge_human_metric_strict_en.jsonl | jsonl | True | 2202 | merged_human_metric | ["eval", "model", "principle", "question", "reason", "response", "score"] |
| merge_human_metric_strict_zh.jsonl | jsonl | True | 2585 | merged_human_metric | ["eval", "model", "principle", "question", "reason", "response", "score"] |
| merge_model_metric.jsonl | jsonl | True | 6661 | llm_judge_output | ["eval", "model", "principle", "question", "reason", "response", "score", "score_mean"] |
| metrics_map.json | json | True | 9 | script_or_config | ["__key__", "__value__"] |
| results_merge.jsonl | jsonl | True | 5536 | merged_human_metric | ["answer", "evaluate", "levels", "metric", "model", "question", "task"] |
| sampled_merge_50_new.json | json | True | 6000 | synthetic_or_augmented | ["input", "instruction", "output"] |
| sampled_merge_50_new_swift.json | json | True | 6000 | synthetic_or_augmented | ["messages", "solution"] |

## All Inventoried Files

| file_path | file_type | exists | num_rows_or_records | likely_role | sample_keys |
| --- | --- | --- | --- | --- | --- |
| 5-grades.py | py | True |  | script_or_config | [] |
| 5-grades/1-shot_cases_zh.json | json | True | 12 | unknown | ["__key__", "generated_responses", "principle", "question"] |
| 5-grades/5_50_metric_v3_questions_en.json | json | True | 12 | unknown | ["__key__", "__value__"] |
| 5-grades/5_50_metric_v3_questions_zh.json | json | True | 12 | unknown | ["__key__", "__value__"] |
| 5-grades/5_50_metric_v3_questions_zh_test.json | json | True | 2 | unknown | ["__key__", "__value__"] |
| 5-grades/5_groupby_metric_v3_eval_en.jsonl | jsonl | True | 2785 | llm_judge_output | ["model", "principle", "question", "reason", "response", "score"] |
| 5-grades/5_groupby_metric_v3_eval_zh.jsonl | jsonl | True | 2805 | llm_judge_output | ["model", "principle", "question", "reason", "response", "score"] |
| 5-grades/5_human_1.jsonl | jsonl | True | 3366 | human_annotation | ["model", "principle", "question", "reason", "response", "score"] |
| 5-grades/5_human_2.jsonl | jsonl | True | 3366 | human_annotation | ["model", "principle", "question", "reason", "response", "score"] |
| 5-grades/5_human_3.jsonl | jsonl | True | 3366 | human_annotation | ["model", "principle", "question", "reason", "response", "score"] |
| 5-grades/5_merge_human_metric_en.jsonl | jsonl | True | 4031 | merged_human_metric | ["eval", "model", "principle", "question", "reason", "response", "score"] |
| 5-grades/5_merge_human_metric_zh.jsonl | jsonl | True | 4007 | merged_human_metric | ["eval", "model", "principle", "question", "reason", "response", "score"] |
| 5-grades/5_metrics_en.json | json | True | 12 | unknown | ["__key__", "description", "rules"] |
| 5-grades/5_metrics_zh.json | json | True | 12 | unknown | ["__key__", "description", "rules"] |
| 5-grades/Untitled-1.json | json | True | 3 | unknown | [] |
| 5-grades/example.jsonl | jsonl | True | 0 | unknown | [] |
| 5-grades/example_zh.jsonl | jsonl | True | 75 | unknown | ["eval", "model", "principle", "question", "reason", "response", "score"] |
| 5-grades/extract_example.py | py | True |  | script_or_config | [] |
| 5-grades/get_example.py | py | True |  | script_or_config | [] |
| 5-grades/judge_v3_questions_zh.jsonl | jsonl | True | 267 | llm_judge_output | [] |
| 5-grades/sample_questions.py | py | True |  | script_or_config | [] |
| EduBench.pdf | pdf | True |  | official_raw | [] |
| EduBench.zip | zip | True | 12 | official_raw | [".bib", ".bst", ".md", ".pdf", ".sty", ".tex", ".txt", "<no_ext>"] |
| Untitled-1.py | py | True |  | script_or_config | [] |
| analysis_outputs/calibration_curve.csv | csv | True |  | unknown | [] |
| analysis_outputs/easiest_questions.csv | csv | True |  | unknown | [] |
| analysis_outputs/evaluator_generator_affinity.csv | csv | True |  | llm_judge_output | [] |
| analysis_outputs/evaluator_summary.csv | csv | True |  | llm_judge_output | [] |
| analysis_outputs/figures/calibration_curves.pdf | pdf | True |  | unknown | [] |
| analysis_outputs/figures/calibration_curves_en.pdf | pdf | True |  | unknown | [] |
| analysis_outputs/figures/calibration_curves_zh.pdf | pdf | True |  | unknown | [] |
| analysis_outputs/figures/evaluator_affinity_barplots.pdf | pdf | True |  | llm_judge_output | [] |
| analysis_outputs/figures/evaluator_affinity_barplots_en.pdf | pdf | True |  | llm_judge_output | [] |
| analysis_outputs/figures/evaluator_affinity_barplots_zh.pdf | pdf | True |  | llm_judge_output | [] |
| analysis_outputs/figures/metric_evaluator_mae_heatmap_top12.pdf | pdf | True |  | llm_judge_output | [] |
| analysis_outputs/figures/metric_evaluator_mae_heatmap_top12_en.pdf | pdf | True |  | llm_judge_output | [] |
| analysis_outputs/figures/metric_evaluator_mae_heatmap_top12_zh.pdf | pdf | True |  | llm_judge_output | [] |
| analysis_outputs/figures/plot_edubench_figures.py | py | True |  | script_or_config | [] |
| analysis_outputs/figures/task_evaluator_bias_heatmap.pdf | pdf | True |  | llm_judge_output | [] |
| analysis_outputs/figures/task_evaluator_bias_heatmap_en.pdf | pdf | True |  | llm_judge_output | [] |
| analysis_outputs/figures/task_evaluator_bias_heatmap_zh.pdf | pdf | True |  | llm_judge_output | [] |
| analysis_outputs/figures/task_evaluator_mae_heatmap.pdf | pdf | True |  | llm_judge_output | [] |
| analysis_outputs/figures/task_evaluator_mae_heatmap_en.pdf | pdf | True |  | llm_judge_output | [] |
| analysis_outputs/figures/task_evaluator_mae_heatmap_zh.pdf | pdf | True |  | llm_judge_output | [] |
| analysis_outputs/hardest_questions.csv | csv | True |  | unknown | [] |
| analysis_outputs/language_task_table.csv | csv | True |  | unknown | [] |
| analysis_outputs/manifest.json | json | True | 4 | unknown | ["__key__", "__value__"] |
| analysis_outputs/metric_evaluator_table.csv | csv | True |  | llm_judge_output | [] |
| analysis_outputs/model_summary.csv | csv | True |  | unknown | [] |
| analysis_outputs/task_evaluator_table.csv | csv | True |  | llm_judge_output | [] |
| analysis_outputs/task_model_table.csv | csv | True |  | unknown | [] |
| categories/analyse1.py | py | True |  | script_or_config | [] |
| categories/analyse2.py | py | True |  | script_or_config | [] |
| categories/analyse3.py | py | True |  | script_or_config | [] |
| categories/category.json | json | True | 18 | unknown | ["'临床医学", "'体育教育学", "'作物科学", "'公共管理学", "'军事学", "'初中", "'化学", "'博士", "'历史", "'历史学", "'商业管理学", "'地理", "'基础医学", "'大学", "'小学", "'应用经济学", "'心理学", "'数学", "'文学与艺术", "'普通教育学", "'水产养殖", ... |
| categories/category_merge.json | json | True | 16 | unknown | ["'higher education", "'k12 level", "__key__"] |
| categories/category_merge_1.json | json | True | 4 | unknown | ["__key__", "higher education", "k12 level"] |
| categories/category_merge_2.json | json | True | 14 | unknown | ["'higher education", "'k12 level", "__key__"] |
| categories/category_no_design.json | json | True | 16 | unknown | ["'临床医学", "'体育教育学", "'作物科学", "'公共管理学", "'军事学", "'化学", "'历史", "'历史学", "'商业管理学", "'地理", "'基础医学", "'应用经济学", "'心理学", "'数学", "'文学与艺术", "'普通教育学", "'水产养殖", "'法学", "'物理", "'物理学", "'理论经济... |
| categories/category_reorganized.json | json | True | 14 | unknown | ["__key__", "初中", "博士", "大学", "小学", "硕士", "高中"] |
| correlation-testset/analysis_edubench.py | py | True |  | script_or_config | [] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_EduBenchEvaluator.json | json | True | 14 | llm_judge_output | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_deepseek-r1.json | json | True | 14 | llm_judge_output | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_deepseek-v3.json | json | True | 14 | llm_judge_output | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_gpt-4o.json | json | True | 14 | llm_judge_output | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_1.json | json | True | 14 | llm_judge_output | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_2.json | json | True | 14 | llm_judge_output | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_3.json | json | True | 14 | llm_judge_output | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_human_mean.json | json | True | 14 | llm_judge_output | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/EduBenchEvaluator_qwq-plus.json | json | True | 14 | llm_judge_output | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_EduBenchEvaluator.json | json | True | 14 | llm_judge_output | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_deepseek-r1.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_deepseek-v3.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_gpt-4o.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human_1.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human_2.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human_3.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_human_mean.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-r1_qwq-plus.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_EduBenchEvaluator.json | json | True | 14 | llm_judge_output | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_deepseek-r1.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_deepseek-v3.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_gpt-4o.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human_1.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human_2.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human_3.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_human_mean.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/deepseek-v3_qwq-plus.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_EduBenchEvaluator.json | json | True | 14 | llm_judge_output | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_deepseek-r1.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_deepseek-v3.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_gpt-4o.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_1.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_2.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_3.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_human_mean.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/gpt-4o_qwq-plus.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_EduBenchEvaluator.json | json | True | 14 | llm_judge_output | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_deepseek-r1.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_deepseek-v3.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_gpt-4o.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_1.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_2.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_3.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_human_mean.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_1_qwq-plus.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_EduBenchEvaluator.json | json | True | 14 | llm_judge_output | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_deepseek-r1.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_deepseek-v3.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_gpt-4o.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_1.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_2.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_3.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_human_mean.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_2_qwq-plus.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_EduBenchEvaluator.json | json | True | 14 | llm_judge_output | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_deepseek-r1.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_deepseek-v3.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_gpt-4o.json | json | True | 14 | unknown | ["__key__", "__value__"] |
| correlation-testset/corr_res_kendall_split_and_fill/human_3_human_1.json | json | True | 14 | unknown | ["__key__", "__value__"] |

_Showing 120 of 466 rows._
