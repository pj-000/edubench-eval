# Exp 0.1 Report: EduBench Reference Alignment and Data Hardening

## 1. Purpose

Exp 0.1 hardens the existing Exp 0 artifacts against three source-of-truth layers: the local PDF audit corpus, official EduBench paper definitions, and the official EduBench GitHub repository. It does not train models, call APIs, or use GPU.

## 2. Source-of-truth hierarchy

1. PDF audit corpus: 5536 generated responses / scored items, 5 generator models, 12 dimensions, 25 subjects, 6 education levels, 2 languages, and 3318/2218 evaluator-vs-human split at question-answer-metric triple granularity.
2. Official EduBench paper: full benchmark has 9 major educational scenarios, 4000+ educational contexts, 18821 data points, 500 sampled human/LLM-evaluated queries, and 12 evaluation aspects.
3. Official EduBench GitHub: canonical 9 scenarios, 12 metrics, and `data/all_data` official data structure when available.

## 3. Official EduBench full data vs PDF audit subset

Official EduBench full data is not the same artifact as the local 5536-row audit corpus. The local dataset is named `edubench_audit_human_scored_subset`; it is a local derived merged human-scored subset from `results_merge.jsonl`, not the full official EduBench dataset. Downstream Exp1 evaluator training/testing should use this audit corpus. Synthetic augmentation or full official EduBench data can be used in later experiments only if kept out of the main human-labeled test set.

## 4. Source Inventory Summary

| likely_role | num_files |
| --- | --- |
| unknown | 294 |
| llm_judge_output | 71 |
| script_or_config | 69 |
| synthetic_or_augmented | 12 |
| official_raw | 8 |
| human_annotation | 6 |
| merged_human_metric | 6 |

## 5. Local source selection

`results_merge.jsonl` is used as the primary local source because each row already represents a scored item and contains `question`, `answer`, `metric`, `task`, generator `model`, plus `evaluate.human_1`, `evaluate.human_2`, and `evaluate.human_3`. `report/results_merge_enriched.jsonl` is used only to recover local enriched audit metadata such as subject, education level, language, and original held-out flag. Synthetic sampled files are excluded.

## 6. Official source inventory

| source_path | source_origin | file_role | num_records | contains_question | contains_answer | contains_metric | contains_task_or_scenario |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EduBench.zip | zip | unknown | 12 | False | False | False | False |
| README.md | official_github_clone | readme | 0 | False | False | False | False |
| code/readme.md | official_github_clone | readme | 0 | False | False | False | False |
| data/all_data/en_data/AG.jsonl | official_github_clone | en_data | 1042 | True | True | False | False |
| data/all_data/en_data/EC.jsonl | official_github_clone | en_data | 1301 | True | True | False | False |
| data/all_data/en_data/ES.jsonl | official_github_clone | en_data | 1061 | True | True | False | False |
| data/all_data/en_data/IP.jsonl | official_github_clone | en_data | 1301 | True | True | False | False |
| data/all_data/en_data/PCC.jsonl | official_github_clone | en_data | 252 | True | True | False | False |
| data/all_data/en_data/PLS.jsonl | official_github_clone | en_data | 448 | True | True | False | True |
| data/all_data/en_data/Q&A.jsonl | official_github_clone | en_data | 1285 | True | True | False | False |
| data/all_data/en_data/QG.jsonl | official_github_clone | en_data | 1288 | True | True | False | False |
| data/all_data/en_data/TMG.jsonl | official_github_clone | en_data | 1185 | True | True | False | False |
| data/all_data/sampled_data/en_data_sampled.jsonl | official_github_clone | en_data | 99 | True | True | True | True |
| data/all_data/sampled_data/zh_data_sampled.jsonl | official_github_clone | zh_data | 99 | True | True | True | True |
| data/all_data/zh_data/AG.jsonl | official_github_clone | zh_data | 931 | True | True | False | False |
| data/all_data/zh_data/EC.jsonl | official_github_clone | zh_data | 620 | True | True | False | False |
| data/all_data/zh_data/IP.jsonl | official_github_clone | zh_data | 1342 | True | True | False | False |
| data/all_data/zh_data/PCC.jsonl | official_github_clone | zh_data | 568 | True | True | False | False |
| data/all_data/zh_data/PLS.jsonl | official_github_clone | zh_data | 348 | True | True | False | True |
| data/all_data/zh_data/Q&A.jsonl | official_github_clone | zh_data | 1306 | True | True | False | False |
| data/all_data/zh_data/QG.jsonl | official_github_clone | zh_data | 1343 | True | True | False | False |
| data/all_data/zh_data/TMG.jsonl | official_github_clone | zh_data | 1335 | True | True | False | False |
| data/readme.md | official_github_clone | readme | 0 | False | False | False | False |

## 7. Metric/scenario alignment

Canonical metric count: 12/12; canonical scenario count: 9/9. Mapping tables are written to `tables/metric_mapping.csv` and `tables/scenario_mapping.csv`.

| field | value |
| --- | --- |
| num_unmapped_metric_error | 0 |
| num_unmapped_metric_info | 11 |
| num_unmapped_scenario_error | 0 |
| num_unmapped_scenario_info | 0 |

## 8. Subject alignment status

Canonical subject count: 25/25. Subject metadata is recovered from the local enriched audit file when available. Subject-level analysis can be used as primary thesis evidence only after confirming this local enriched subject annotation; otherwise treat it as audit metadata.

## 9. Score scale audit

Raw human scores in the main dataset are detected as: {'1-5': 5536}. Because the current `results_merge.jsonl` human scores are already on a 1-5 scale, `human_1_5`/`human_2_5`/`human_3_5` equal the raw values and `label_5` is `round(human_mean_5)` clipped to 1-5. The 1-10 mapping compatible with `5-grades.py` is still implemented for any future 1-10 source: 1-2->1, 3-4->2, 5-6->3, 7-8->4, 9-10->5.

## 10. Main dataset statistics

| item | observed | reference |
| --- | --- | --- |
| dataset_name | edubench_audit_human_scored_subset | PDF audit subset |
| total_scored_items | 5536 | 5536 |
| unique_triple_key | 5423 | question-answer-metric granularity |
| unique_question_key | 197 | question robustness split |
| unique_answer_key | 963 | leakage diagnostic |
| generator_models | 5 | 5 |
| canonical_metrics | 12 | 12 |
| canonical_scenarios | 9 | 9 |
| canonical_subjects | 25 | 25 |
| education_levels | 6 | 6 |
| languages | ["en", "zh"] | English / Chinese |
| paper train_pool/test | targeted by paper_like_triple_seed42 | 3318 / 2218 |

### label_5 distribution

| value | count | pct |
| --- | --- | --- |
| 5 | 2927 | 52.87% |
| 4 | 1903 | 34.38% |
| 3 | 507 | 9.16% |
| 2 | 113 | 2.04% |
| 1 | 86 | 1.55% |

## 11. Split reference check

`paper_like_triple_seed42` split source: `original_heldout_flag_repaired_for_triple_isolation`.

| split_name | split | split_source | target_rows | num_rows | train_pool_rows | metric_count | scenario_count | subject_count | education_level_count | language_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_like_triple_seed42 | train | original_heldout_flag_repaired_for_triple_isolation | 2654 | 2654 |  | 12 | 9 | 25 | 6 | 2 |
| paper_like_triple_seed42 | dev | original_heldout_flag_repaired_for_triple_isolation | 664 | 664 |  | 12 | 9 | 25 | 6 | 2 |
| paper_like_triple_seed42 | test | original_heldout_flag_repaired_for_triple_isolation | 2218 | 2218 | 3318 | 12 | 8 | 25 | 6 | 2 |
| question_seed42 | train | deterministic_reconstructed_question_seed42 | 3322 | 3326 |  | 12 | 9 | 24 | 6 | 2 |
| question_seed42 | dev | deterministic_reconstructed_question_seed42 | 1107 | 1107 |  | 12 | 9 | 23 | 5 | 2 |
| question_seed42 | test | deterministic_reconstructed_question_seed42 | 1107 | 1103 | 4433 | 12 | 9 | 18 | 6 | 2 |

## 12. Leakage check

Leakage status: **WARNING**. See `leakage_report.md`, `tables/leakage_summary.csv`, and `tables/leakage_details.csv` for the exact checks.

## 13. Low-score distribution

Labels 1/2 account for 199 of 5536 scored items (3.59%).

## 14. Figures generated

- `figures/fig00_annotator_score_distribution.png`
- `figures/fig00_education_level_distribution.png`
- `figures/fig00_generator_model_distribution.png`
- `figures/fig00_human_annotator_agreement.png`
- `figures/fig00_language_distribution.png`
- `figures/fig00_metric_distribution.png`
- `figures/fig00_missingness_heatmap.png`
- `figures/fig00_raw_score_distribution.png`
- `figures/fig00_reference_flow.png`
- `figures/fig00_scenario_distribution.png`
- `figures/fig00_score_distribution.png`
- `figures/fig00_score_mapping_audit.png`
- `figures/fig00_split_coverage_heatmap_metric.png`
- `figures/fig00_split_coverage_heatmap_scenario.png`
- `figures/fig00_split_score_distribution_paper_like_triple.png`
- `figures/fig00_subject_alignment_status.png`
- `figures/fig00_subject_distribution_top25.png`

## 15. Remaining warnings before Exp1

- Sanity status: PASS.
- Leakage status: WARNING.
- Synthetic sampled overlap is reported only as a future Exp6 augmentation risk; synthetic rows are excluded from the main dataset.
- Confirm that local enriched subject annotations are acceptable before treating subject-level analysis as primary evidence.
