# Exp6 Review Package

Can Exp6 training start? **NO**

## Recommended Usable Sources

| source_file | candidate_rows | target_label_5_rows | low_score_rows | leakage_risk | recommended_use |
| --- | --- | --- | --- | --- | --- |
| deepseek_output/processed_excel_data_1.jsonl | 659 | 659 | 8 | LOW | POSSIBLE_FILTERED_TRAIN_ONLY_AFTER_MANUAL_CONFIRMATION |
| deepseek_output/processed_excel_data_1_en.jsonl | 331 | 331 | 5 | LOW | POSSIBLE_FILTERED_TRAIN_ONLY_AFTER_MANUAL_CONFIRMATION |
| deepseek_output/processed_excel_data_1_zh.jsonl | 328 | 328 | 3 | LOW | POSSIBLE_FILTERED_TRAIN_ONLY_AFTER_MANUAL_CONFIRMATION |

## Blocked Sources

| source_file | likely_role | candidate_rows | blocked_reasons |
| --- | --- | --- | --- |
|  | unknown | 6 | no target_label_5; no low-score target_label_5 rows |
| deepseek-r1_merged.jsonl | model_judge_output | 990 | exact dev/test overlap detected; model/judge output, not human label;... |
| deepseek_output/deepseek_generated_20251124_204411.jsonl | synthetic_candidate | 164 | no target_label_5; no low-score target_label_5 rows |
| deepseek_output/en_judge_20251109_221657.jsonl | model_judge_output | 173 | model/judge output, not human label; no target_label_5; no low-score ... |
| deepseek_output/judge-1-process.jsonl | model_judge_output | 189 | model/judge output, not human label; no target_label_5; no low-score ... |
| deepseek_output/judge-2-process.jsonl | model_judge_output | 186 | model/judge output, not human label; no target_label_5; no low-score ... |
| deepseek_output/processed_excel_data_2.jsonl | model_judge_output | 659 | model/judge output, not human label |
| deepseek_output/processed_excel_data_2_en.jsonl | model_judge_output | 331 | model/judge output, not human label |
| deepseek_output/processed_excel_data_2_zh.jsonl | model_judge_output | 328 | model/judge output, not human label |
| deepseek_output/unique_questions.jsonl | unknown | 25 | no target_label_5; no low-score target_label_5 rows |
| deepseek_output/zh_judge_20251109_223622.jsonl | model_judge_output | 234 | model/judge output, not human label; no target_label_5; no low-score ... |
| edu-data-synthesis-main/data/criteria/metrics_old.json | synthetic_candidate | 3 | no target_label_5; no low-score target_label_5 rows |
| edu-data-synthesis-main/data/eval_data/eval_samples.jsonl | synthetic_candidate | 17400 | exact dev/test overlap detected |
| edu-data-synthesis-main/data/eval_data/train_eval_data.jsonl | model_judge_output | 10098 | exact dev/test overlap detected; model/judge output, not human label |
| edu-data-synthesis-main/data/eval_data/val_eval_data.jsonl | model_judge_output | 6732 | exact dev/test overlap detected; model/judge output, not human label |
| edu-data-synthesis-main/data/zh/cjeval.jsonl | sampled_augmented | 26136 | no target_label_5; no low-score target_label_5 rows |
| edu-data-synthesis-main/data/zh/gaokao-bench.jsonl | sampled_augmented | 3126 | no target_label_5; no low-score target_label_5 rows |
| edu-data-synthesis-main/init_workflows/init_evaluation_workflow_1_1.json | synthetic_candidate | 1 | no target_label_5; no low-score target_label_5 rows |
| edu-data-synthesis-main/test_workflow.json | synthetic_candidate | 1 | no target_label_5; no low-score target_label_5 rows |
| groupby_metric_qwq_eval_en.jsonl | model_judge_output | 2709 | exact dev/test overlap detected; model/judge output, not human label;... |
| groupby_metric_qwq_eval_zh.jsonl | model_judge_output | 2732 | exact dev/test overlap detected; model/judge output, not human label;... |
| groupby_metric_r1_eval_en.jsonl | model_judge_output | 2767 | exact dev/test overlap detected; model/judge output, not human label;... |
| groupby_metric_r1_eval_zh.jsonl | model_judge_output | 2624 | exact dev/test overlap detected; model/judge output, not human label;... |
| groupby_metric_v3_eval_en.jsonl | model_judge_output | 2785 | exact dev/test overlap detected; model/judge output, not human label;... |
| groupby_metric_v3_eval_zh.jsonl | model_judge_output | 2805 | exact dev/test overlap detected; model/judge output, not human label;... |
| human_sampled_eval_sft_criteria_test.json | sampled_augmented | 3274 | exact dev/test overlap detected; test-style human-sampled SFT file |
| merge_model_metric.jsonl | model_judge_output | 6661 | exact dev/test overlap detected; model/judge output, not human label;... |
| sampled_merge_50_new.json | sampled_augmented | 6000 | exact dev/test overlap detected; required HIGH risk sampled_merge source |
| sampled_merge_50_new_swift.json | sampled_augmented | 6000 | required HIGH risk sampled_merge source |

## Leakage Status

Dev/test leakage source count: **13**.

| source_file | total_candidates | leakage_risk | notes |
| --- | --- | --- | --- |
| deepseek-r1_merged.jsonl | 990 | BLOCKED | dev/test overlap found |
| edu-data-synthesis-main/data/eval_data/eval_samples.jsonl | 17400 | BLOCKED | dev/test overlap found |
| edu-data-synthesis-main/data/eval_data/train_eval_data.jsonl | 10098 | BLOCKED | dev/test overlap found |
| edu-data-synthesis-main/data/eval_data/val_eval_data.jsonl | 6732 | BLOCKED | dev/test overlap found |
| groupby_metric_qwq_eval_en.jsonl | 2709 | BLOCKED | dev/test overlap found |
| groupby_metric_qwq_eval_zh.jsonl | 2732 | BLOCKED | dev/test overlap found |
| groupby_metric_r1_eval_en.jsonl | 2767 | BLOCKED | dev/test overlap found |
| groupby_metric_r1_eval_zh.jsonl | 2624 | BLOCKED | dev/test overlap found |
| groupby_metric_v3_eval_en.jsonl | 2785 | BLOCKED | dev/test overlap found |
| groupby_metric_v3_eval_zh.jsonl | 2805 | BLOCKED | dev/test overlap found |
| human_sampled_eval_sft_criteria_test.json | 3274 | BLOCKED | dev/test overlap found |
| merge_model_metric.jsonl | 6661 | BLOCKED | dev/test overlap found |
| sampled_merge_50_new.json | 6000 | BLOCKED | dev/test overlap found |

## Label Reliability Status

Labels are **not accepted as human labels by default**. Parsed `target_label_5` exists for
**75109** rows, but provenance is model/SFT/synthetic unless manually confirmed.

## Error Type Availability

Rows with `error_type`: **0**.

## Proposed First Training Matrix

1. E6-H0: human-only reference, unchanged dev/test.
2. E6-F1: human + filtered low-score synthetic, train only.
3. E6-F2: human + filtered low-score synthetic with per-metric and per-language caps.
4. E6-D1: synthetic-only diagnostic, never as final replacement.

## Required Manual Confirmations

- Confirm which non-judge sources have trustworthy label provenance.
- Remove or quarantine every dev/test overlap listed in leakage details.
- Confirm `sampled_merge_*` labels and source-question provenance before any train-only use.
- Confirm no synthetic rows are introduced into dev/test.
- Confirm synthetic mix ratio and per-metric caps before training.
