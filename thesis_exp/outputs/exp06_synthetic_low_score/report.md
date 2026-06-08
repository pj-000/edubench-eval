# Exp6 Synthetic Low-Score Data Inventory / Audit

## Scope

This audit scans existing synthetic, sampled, augmented, model-judge, and synthesis-script artifacts
only. It does not train models, call APIs, generate synthetic data, modify Exp0-Exp5 results, or add
synthetic rows to train/dev/test.

## Source Inventory

Inventory rows: **96**

| likely_role | num_sources |
| --- | --- |
| generation_script | 51 |
| model_judge_output | 23 |
| synthetic_candidate | 10 |
| unknown | 7 |
| sampled_augmented | 5 |

| risk_level | num_sources |
| --- | --- |
| BLOCKED | 74 |
| HIGH | 21 |
| MEDIUM | 1 |

## Normalized Candidate Overview

Normalized audit candidate rows: **106457**

Rows with `target_label_5`: **75109**

Low-score candidate rows (`target_label_5` in 1/2): **6687**

Rows with `error_type`: **0**

## Required Questions

1. What synthetic / sampled sources exist?

The requested roots were scanned recursively where applicable. See
`tables/synthetic_source_inventory.csv` and `tables/synthetic_schema_profile.csv`. Main categories
are sampled SFT wrappers (`sampled_merge_*`, `human_sampled_eval_sft_criteria_test.json`),
`edu-data-synthesis-main` synthesis/eval data and scripts, `deepseek_output` / `qwen_output`, and
model judge outputs such as `groupby_metric_*`, `merge_model_metric.jsonl`, and
`deepseek-r1_merged.jsonl`.

2. Which can be used for Exp6?

Only sources listed as `POSSIBLE_FILTERED_TRAIN_ONLY_AFTER_MANUAL_CONFIRMATION` in
`tables/synthetic_filter_recommendation.csv` are possible candidates, and only for train-side
augmentation after manual label provenance review. Synthetic data must never be used as dev/test.

| source_file | candidate_rows | target_label_5_rows | low_score_rows | leakage_risk | recommended_use |
| --- | --- | --- | --- | --- | --- |
| deepseek_output/processed_excel_data_1.jsonl | 659 | 659 | 8 | LOW | POSSIBLE_FILTERED_TRAIN_ONLY_AFTER_MANUAL_CONFIRMATION |
| deepseek_output/processed_excel_data_1_en.jsonl | 331 | 331 | 5 | LOW | POSSIBLE_FILTERED_TRAIN_ONLY_AFTER_MANUAL_CONFIRMATION |
| deepseek_output/processed_excel_data_1_zh.jsonl | 328 | 328 | 3 | LOW | POSSIBLE_FILTERED_TRAIN_ONLY_AFTER_MANUAL_CONFIRMATION |

3. Which cannot be used directly?

`groupby_metric_*_eval_*`, `merge_model_metric.jsonl`, and `deepseek-r1_merged.jsonl` are
model/judge outputs, not human labels. `sampled_merge_50_new.json` and
`sampled_merge_50_new_swift.json` are default HIGH risk. `human_sampled_eval_sft_criteria_test.json`
is treated as a test-style SFT sample and is not a direct train/dev/test source.

| source_file | likely_role | candidate_rows | low_score_rows | recommended_use | blocked_reasons |
| --- | --- | --- | --- | --- | --- |
|  | unknown | 6 | 0 | BLOCKED_OR_REVIEW_ONLY | no target_label_5; no low-score target_label_5 rows |
| deepseek-r1_merged.jsonl | model_judge_output | 990 | 0 | BLOCKED_OR_REVIEW_ONLY | exact dev/test overlap detected; model/judge output, not human label;... |
| deepseek_output/deepseek_generated_20251124_204411.jsonl | synthetic_candidate | 164 | 0 | BLOCKED_OR_REVIEW_ONLY | no target_label_5; no low-score target_label_5 rows |
| deepseek_output/en_judge_20251109_221657.jsonl | model_judge_output | 173 | 0 | BLOCKED_OR_REVIEW_ONLY | model/judge output, not human label; no target_label_5; no low-score ... |
| deepseek_output/judge-1-process.jsonl | model_judge_output | 189 | 0 | BLOCKED_OR_REVIEW_ONLY | model/judge output, not human label; no target_label_5; no low-score ... |
| deepseek_output/judge-2-process.jsonl | model_judge_output | 186 | 0 | BLOCKED_OR_REVIEW_ONLY | model/judge output, not human label; no target_label_5; no low-score ... |
| deepseek_output/processed_excel_data_2.jsonl | model_judge_output | 659 | 8 | BLOCKED_OR_REVIEW_ONLY | model/judge output, not human label |
| deepseek_output/processed_excel_data_2_en.jsonl | model_judge_output | 331 | 5 | BLOCKED_OR_REVIEW_ONLY | model/judge output, not human label |
| deepseek_output/processed_excel_data_2_zh.jsonl | model_judge_output | 328 | 3 | BLOCKED_OR_REVIEW_ONLY | model/judge output, not human label |
| deepseek_output/unique_questions.jsonl | unknown | 25 | 0 | BLOCKED_OR_REVIEW_ONLY | no target_label_5; no low-score target_label_5 rows |
| deepseek_output/zh_judge_20251109_223622.jsonl | model_judge_output | 234 | 0 | BLOCKED_OR_REVIEW_ONLY | model/judge output, not human label; no target_label_5; no low-score ... |
| edu-data-synthesis-main/data/criteria/metrics_old.json | synthetic_candidate | 3 | 0 | BLOCKED_OR_REVIEW_ONLY | no target_label_5; no low-score target_label_5 rows |
| edu-data-synthesis-main/data/eval_data/eval_samples.jsonl | synthetic_candidate | 17400 | 635 | BLOCKED_OR_REVIEW_ONLY | exact dev/test overlap detected |
| edu-data-synthesis-main/data/eval_data/train_eval_data.jsonl | model_judge_output | 10098 | 428 | BLOCKED_OR_REVIEW_ONLY | exact dev/test overlap detected; model/judge output, not human label |
| edu-data-synthesis-main/data/eval_data/val_eval_data.jsonl | model_judge_output | 6732 | 207 | BLOCKED_OR_REVIEW_ONLY | exact dev/test overlap detected; model/judge output, not human label |
| edu-data-synthesis-main/data/zh/cjeval.jsonl | sampled_augmented | 26136 | 0 | BLOCKED_OR_REVIEW_ONLY | no target_label_5; no low-score target_label_5 rows |
| edu-data-synthesis-main/data/zh/gaokao-bench.jsonl | sampled_augmented | 3126 | 0 | BLOCKED_OR_REVIEW_ONLY | no target_label_5; no low-score target_label_5 rows |
| edu-data-synthesis-main/init_workflows/init_evaluation_workflow_1_1.json | synthetic_candidate | 1 | 0 | BLOCKED_OR_REVIEW_ONLY | no target_label_5; no low-score target_label_5 rows |
| edu-data-synthesis-main/test_workflow.json | synthetic_candidate | 1 | 0 | BLOCKED_OR_REVIEW_ONLY | no target_label_5; no low-score target_label_5 rows |
| groupby_metric_qwq_eval_en.jsonl | model_judge_output | 2709 | 91 | BLOCKED_OR_REVIEW_ONLY | exact dev/test overlap detected; model/judge output, not human label;... |
| groupby_metric_qwq_eval_zh.jsonl | model_judge_output | 2732 | 59 | BLOCKED_OR_REVIEW_ONLY | exact dev/test overlap detected; model/judge output, not human label;... |
| groupby_metric_r1_eval_en.jsonl | model_judge_output | 2767 | 121 | BLOCKED_OR_REVIEW_ONLY | exact dev/test overlap detected; model/judge output, not human label;... |
| groupby_metric_r1_eval_zh.jsonl | model_judge_output | 2624 | 77 | BLOCKED_OR_REVIEW_ONLY | exact dev/test overlap detected; model/judge output, not human label;... |
| groupby_metric_v3_eval_en.jsonl | model_judge_output | 2785 | 79 | BLOCKED_OR_REVIEW_ONLY | exact dev/test overlap detected; model/judge output, not human label;... |
| groupby_metric_v3_eval_zh.jsonl | model_judge_output | 2805 | 2 | BLOCKED_OR_REVIEW_ONLY | exact dev/test overlap detected; model/judge output, not human label;... |
| human_sampled_eval_sft_criteria_test.json | sampled_augmented | 3274 | 110 | BLOCKED_OR_REVIEW_ONLY | exact dev/test overlap detected; test-style human-sampled SFT file |
| merge_model_metric.jsonl | model_judge_output | 6661 | 46 | BLOCKED_OR_REVIEW_ONLY | exact dev/test overlap detected; model/judge output, not human label;... |
| sampled_merge_50_new.json | sampled_augmented | 6000 | 2400 | HIGH_RISK_REVIEW_ONLY | exact dev/test overlap detected; required HIGH risk sampled_merge source |
| sampled_merge_50_new_swift.json | sampled_augmented | 6000 | 2400 | HIGH_RISK_REVIEW_ONLY | required HIGH risk sampled_merge source |

4. Is there dev/test leakage risk?

Dev/test leakage source count: **13**. Exact details are in
`tables/synthetic_leakage_details.csv`. Any source with dev/test overlap is blocked until manual
review removes or quarantines the overlapping rows.

| source_file | total_candidates | question_key_in_dev | question_key_in_test | triple_key_in_dev | triple_key_in_test | normalized_qa_in_dev | normalized_qa_in_test | leakage_risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-r1_merged.jsonl | 990 | 925 | 820 | 0 | 0 | 506 | 757 | BLOCKED |
| edu-data-synthesis-main/data/eval_data/eval_samples.jsonl | 17400 | 16370 | 14565 | 2050 | 6869 | 9226 | 13698 | BLOCKED |
| edu-data-synthesis-main/data/eval_data/train_eval_data.jsonl | 10098 | 9540 | 8397 | 1185 | 3894 | 5253 | 7785 | BLOCKED |
| edu-data-synthesis-main/data/eval_data/val_eval_data.jsonl | 6732 | 6360 | 5598 | 795 | 2724 | 3723 | 5367 | BLOCKED |
| groupby_metric_qwq_eval_en.jsonl | 2709 | 2562 | 2306 | 628 | 2184 | 1445 | 2199 | BLOCKED |
| groupby_metric_qwq_eval_zh.jsonl | 2732 | 2598 | 2304 | 323 | 1085 | 1472 | 2130 | BLOCKED |
| groupby_metric_r1_eval_en.jsonl | 2767 | 2618 | 2314 | 656 | 2208 | 1471 | 2207 | BLOCKED |
| groupby_metric_r1_eval_zh.jsonl | 2624 | 2502 | 2240 | 313 | 1049 | 1422 | 2076 | BLOCKED |
| groupby_metric_v3_eval_en.jsonl | 2785 | 2635 | 2340 | 646 | 2212 | 1474 | 2233 | BLOCKED |
| groupby_metric_v3_eval_zh.jsonl | 2805 | 2670 | 2340 | 334 | 1108 | 1513 | 2166 | BLOCKED |
| human_sampled_eval_sft_criteria_test.json | 3274 | 2737 | 2771 | 0 | 0 | 0 | 0 | BLOCKED |
| merge_model_metric.jsonl | 6661 | 6263 | 5911 | 1524 | 5632 | 3450 | 5631 | BLOCKED |
| sampled_merge_50_new.json | 6000 | 2994 | 3069 | 0 | 0 | 0 | 0 | BLOCKED |

5. Is `target_label_5` available?

Yes for rows where an official EduBench metric score could be parsed. These labels are still not
assumed human labels; model/judge/SFT provenance remains recorded in `normalization_status`.

6. Is `error_type` available?

Only sparsely. `error_type` is available for **0** normalized rows.

7. Can we do filtered synthetic?

YES, as a candidate design only: filter to train-only rows, remove all
dev/test overlaps, require `target_label_5`, prefer low-score rows, block judge-only sources, and
manually confirm label provenance.

8. Can we do synthetic-only?

Not as a thesis-quality final model yet. A synthetic-only diagnostic may be run only after manual
confirmation and must still evaluate on unchanged human dev/test; it should be labeled diagnostic,
not comparable as a replacement for human training.

9. Can we do human + synthetic mix?

Potentially yes after filtering and manual confirmation. The human dev/test split must remain
unchanged; synthetic rows can enter train only.

10. Recommended next training matrix

Training should not start until manual confirmations are done. Proposed first matrix after approval:

| run | train data | synthetic filter | dev/test |
| --- | --- | --- | --- |
| E6-H0 | existing human train only | none | unchanged human dev/test |
| E6-F1 | human train + filtered synthetic low-score | no dev/test overlap, `target_label_5` in 1/2, non-judge source | unchanged human dev/test |
| E6-F2 | human train + filtered synthetic low-score + matched metric cap | F1 plus per-metric cap to avoid distribution distortion | unchanged human dev/test |
| E6-D1 | synthetic-only diagnostic | same filter as F1 | unchanged human dev/test |

## Artifact Index

- `thesis_exp/outputs/exp06_synthetic_low_score/tables/synthetic_source_inventory.csv`
- `thesis_exp/outputs/exp06_synthetic_low_score/tables/synthetic_schema_profile.csv`
- `thesis_exp/outputs/exp06_synthetic_low_score/tables/synthetic_candidate_rows.csv`
- `thesis_exp/outputs/exp06_synthetic_low_score/tables/synthetic_leakage_summary.csv`
- `thesis_exp/outputs/exp06_synthetic_low_score/tables/synthetic_filter_recommendation.csv`
