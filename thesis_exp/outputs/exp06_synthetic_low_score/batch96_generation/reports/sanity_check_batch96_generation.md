# Exp6-5 Batch96 Generation Sanity Check

Overall status: **PASS**

| check_name | status | count | notes |
| --- | --- | --- | --- |
| curated_usable_count_16 | PASS | 16 | manual-reviewed mini-batch usable pool |
| mb008_excluded | PASS | 0 | mb008 must not enter low-score curated pool |
| planned_count_96 | PASS | 96 | batch96 planned raw generations |
| label_distribution | PASS | 96 | {'1': 40, '2': 40, '3': 16} |
| language_distribution | PASS | 96 | {'en': 48, 'zh': 48} |
| metric_coverage_12 | PASS | 12 | 12 EduBench metrics if possible |
| error_type_coverage_7 | PASS | 7 | {'reasoning_gap': 12, 'overconfident_wrong': 8, 'scenario_mismatch': ... |
| source_selection_complete | PASS | 96 | selected=96 planned=96 |
| no_dev_test_source_overlap | PASS | 0 | question_seed42/train only; dev/test question forbidden |
| prompt_hardening_status | PASS | 3 | hardened templates include label/error self-checks and target-label g... |
| filter_hardening_status | PASS | 6 | label_plausibility_status, error_type_alignment_status, rubric_failur... |
| api_not_called | PASS | 0 | dry-run only unless EXP6_RUN_GENERATION=1 |
