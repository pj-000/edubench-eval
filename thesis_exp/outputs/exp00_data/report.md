# Exp 0 Report: EduBench Data Audit and Split Construction

## 1. Purpose

This experiment verifies available EduBench-related data sources, builds a normalized human-scored scored-item dataset, creates leakage-aware splits, and generates thesis-ready distribution figures. It does not train models, call APIs, or use GPU.

## 2. Source Inventory Summary

| likely_role | num_files |
| --- | --- |
| unknown | 294 |
| llm_judge_output | 71 |
| script_or_config | 69 |
| synthetic_or_augmented | 12 |
| official_raw | 8 |
| human_annotation | 6 |
| merged_human_metric | 6 |

## 3. Main Dataset Source Selection

`results_merge.jsonl` is used as the main source because each row already represents a scored item and contains `question`, `answer`, `metric`, `task`, generator `model`, plus `evaluate.human_1`, `evaluate.human_2`, and `evaluate.human_3`. The raw `human_1/2/3.jsonl` files are retained as annotation provenance and schema evidence but are not concatenated into the main set because they use a 1-10 scale and would duplicate the merged scored items. Synthetic sampled files are excluded.

## 4. Field Mapping and Canonicalization

Canonical metric count: 12; canonical scenario count: 9. Mapping tables are written to `tables/metric_mapping.csv` and `tables/scenario_mapping.csv`. Unmapped metric rows: 11; unmapped scenario rows: 0.

## 5. Official EduBench Alignment

The canonical references use the official EduBench 9 scenarios and 12 metrics. The processed dataset maps observed raw metric/scenario strings to those canonical names and keeps unmapped values in explicit tables instead of dropping them silently.

## 6. Reference Check

| item | observed | reference |
| --- | --- | --- |
| total_scored_items | 5536 | 5536 |
| unique_triple_key | 5423 | question-answer-metric granularity |
| unique_question_key | 197 | question robustness split |
| unique_answer_key | 963 | leakage diagnostic |
| generator_models | 5 | 5 |
| canonical_metrics | 12 | 12 |
| canonical_scenarios | 9 | 9 |
| canonical_subjects | 21 | 25 |
| education_levels | 6 | 6 |
| languages | ["en", "zh"] | English / Chinese |
| paper train_pool/test | targeted by paper_like_triple_seed42 | 3318 / 2218 |

## 7. Main Dataset Statistics

| value | count | pct |
| --- | --- | --- |
| 5 | 2927 | 52.87% |
| 4 | 1903 | 34.38% |
| 3 | 507 | 9.16% |
| 2 | 113 | 2.04% |
| 1 | 86 | 1.55% |

## 8. Split Statistics

| split_name | split | target_rows | num_rows | pct_rows | unique_triple_key | unique_question_key |
| --- | --- | --- | --- | --- | --- | --- |
| paper_like_triple_seed42 | train | 2654 | 2654 | 0.479408 | 2600 | 197 |
| paper_like_triple_seed42 | dev | 664 | 664 | 0.119942 | 647 | 189 |
| paper_like_triple_seed42 | test | 2218 | 2218 | 0.40065 | 2176 | 197 |
| question_seed42 | train | 3322 | 3326 | 0.600795 | 3233 | 118 |
| question_seed42 | dev | 1107 | 1107 | 0.199964 | 1091 | 40 |
| question_seed42 | test | 1107 | 1103 | 0.199241 | 1099 | 39 |

## 9. Leakage Check Conclusion

Leakage status: **WARNING**. See `leakage_report.md`, `tables/leakage_summary.csv`, and `tables/leakage_details.csv` for the exact checks.

## 10. Low-Score Sample Proportion

Labels 1/2 account for 199 of 5536 scored items (3.59%).

## 11. Recommended Main Split

`paper_like_triple_seed42` is recommended as the main split for evaluator-vs-human comparison because it enforces question-answer-metric triple isolation and matches the paper-like train-pool/test size when the processed total is 5536. Use `question_seed42` as a stricter robustness split.

## 12. Generated Figures

- `figures/fig00_annotator_score_distribution.png`
- `figures/fig00_education_level_distribution.png`
- `figures/fig00_generator_model_distribution.png`
- `figures/fig00_human_annotator_agreement.png`
- `figures/fig00_language_distribution.png`
- `figures/fig00_metric_distribution.png`
- `figures/fig00_missingness_heatmap.png`
- `figures/fig00_raw_score_distribution.png`
- `figures/fig00_scenario_distribution.png`
- `figures/fig00_score_distribution.png`
- `figures/fig00_split_score_distribution_paper_like_triple.png`
- `figures/fig00_subject_distribution_top25.png`
