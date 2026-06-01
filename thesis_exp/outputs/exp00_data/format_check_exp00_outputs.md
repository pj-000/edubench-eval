# Exp 0.1 Formatting Check

Overall status: **PASS**

| artifact | check | status | observed | expected | notes |
| --- | --- | --- | --- | --- | --- |
| /Users/sss/edubench-eval/thesis_exp/configs/reference_contract.yaml | yaml.safe_load | PASS | dict | dict | 89 physical lines |
| /Users/sss/edubench-eval/thesis_exp/configs/reference_contract.yaml | multi-line YAML | PASS | 89 | >=20 | prevents one-line flow-style contract output |
| /Users/sss/edubench-eval/thesis_exp/configs/reference_contract.yaml | official_edubench.expected_scenarios | PASS | ['Question Answering', 'Error Correction', 'Idea Provision', 'Personalized Learning Support', 'Emotional Support', 'Question Generation', 'Automatic Grading', 'Teaching Material Generation', 'Personalized Content Creation'] | list length 9 |  |
| /Users/sss/edubench-eval/thesis_exp/configs/reference_contract.yaml | official_edubench.expected_metrics | PASS | 12 | 12 metrics across groups |  |
| /Users/sss/edubench-eval/thesis_exp/configs/reference_contract.yaml | pdf_audit_corpus.expected_total_scored_items | PASS | 5536 | positive integer |  |
| /Users/sss/edubench-eval/thesis_exp/configs/reference_contract.yaml | pdf_audit_corpus.expected_train_pool_rows | PASS | 3318 | positive integer |  |
| /Users/sss/edubench-eval/thesis_exp/configs/reference_contract.yaml | pdf_audit_corpus.expected_heldout_test_rows | PASS | 2218 | positive integer |  |
| /Users/sss/edubench-eval/thesis_exp/configs/reference_contract.yaml | score_mapping.ten_to_five | PASS | {'1,2': 1, '3,4': 2, '5,6': 3, '7,8': 4, '9,10': 5} | 5 mapping buckets |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/report.md | multi-line markdown | PASS | 172 | >=5 | 6 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/report.md | heading separation | PASS | 0 | 0 | 6 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/report.md | table block format | PASS | 0 | 0 | 6 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/report.md | raw line length | PASS | 227 | <=240 | 6 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/data_card.md | multi-line markdown | PASS | 115 | >=5 | 7 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/data_card.md | heading separation | PASS | 0 | 0 | 7 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/data_card.md | table block format | PASS | 0 | 0 | 7 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/data_card.md | raw line length | PASS | 148 | <=240 | 7 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/review_package.md | multi-line markdown | PASS | 92 | >=5 | 4 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/review_package.md | heading separation | PASS | 0 | 0 | 4 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/review_package.md | table block format | PASS | 0 | 0 | 4 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/review_package.md | raw line length | PASS | 211 | <=240 | 4 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/leakage_report.md | multi-line markdown | PASS | 40 | >=5 | 2 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/leakage_report.md | heading separation | PASS | 0 | 0 | 2 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/leakage_report.md | table block format | PASS | 0 | 0 | 2 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/leakage_report.md | raw line length | PASS | 100 | <=240 | 2 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/official_source_audit.md | multi-line markdown | PASS | 49 | >=5 | 2 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/official_source_audit.md | heading separation | PASS | 0 | 0 | 2 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/official_source_audit.md | table block format | PASS | 0 | 0 | 2 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/official_source_audit.md | raw line length | PASS | 192 | <=240 | 2 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/subject_alignment_report.md | multi-line markdown | PASS | 56 | >=5 | 2 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/subject_alignment_report.md | heading separation | PASS | 0 | 0 | 2 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/subject_alignment_report.md | table block format | PASS | 0 | 0 | 2 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/subject_alignment_report.md | raw line length | PASS | 100 | <=240 | 2 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/split_reference_check.md | multi-line markdown | PASS | 21 | >=5 | 1 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/split_reference_check.md | heading separation | PASS | 0 | 0 | 1 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/split_reference_check.md | table block format | PASS | 0 | 0 | 1 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/split_reference_check.md | raw line length | PASS | 100 | <=240 | 1 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/sanity_check_exp00_reference.md | multi-line markdown | PASS | 26 | >=5 | 1 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/sanity_check_exp00_reference.md | heading separation | PASS | 0 | 0 | 1 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/sanity_check_exp00_reference.md | table block format | PASS | 0 | 0 | 1 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/sanity_check_exp00_reference.md | raw line length | PASS | 219 | <=240 | 1 table block(s) |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/dataset_rows.csv | pandas.read_csv | PASS | 5536 rows / 25 columns | readable CSV | 5537 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/dataset_rows.csv | header row | PASS | 25 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/dataset_rows.csv | one physical line per record | PASS | 5537 | 5537 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/duplicate_triples.csv | pandas.read_csv | PASS | 77 rows / 6 columns | readable CSV | 78 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/duplicate_triples.csv | header row | PASS | 6 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/duplicate_triples.csv | one physical line per record | PASS | 78 | 78 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/format_check_results.csv | pandas.read_csv | PASS | 128 rows / 6 columns | readable CSV | 129 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/format_check_results.csv | header row | PASS | 6 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/format_check_results.csv | one physical line per record | PASS | 129 | 129 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/invalid_or_ambiguous_scores.csv | pandas.read_csv | PASS | 0 rows / 12 columns | readable CSV | 1 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/invalid_or_ambiguous_scores.csv | header row | PASS | 12 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/invalid_or_ambiguous_scores.csv | one physical line per record | PASS | 1 | 1 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/leakage_details.csv | pandas.read_csv | PASS | 63 rows / 17 columns | readable CSV | 64 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/leakage_details.csv | header row | PASS | 17 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/leakage_details.csv | one physical line per record | PASS | 64 | 64 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/leakage_summary.csv | pandas.read_csv | PASS | 18 rows / 5 columns | readable CSV | 19 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/leakage_summary.csv | header row | PASS | 5 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/leakage_summary.csv | one physical line per record | PASS | 19 | 19 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/leakage_summary.csv | required data rows | PASS | 18 | >0 | required Exp0.1 audit CSV |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/metric_mapping.csv | pandas.read_csv | PASS | 239 rows / 10 columns | readable CSV | 240 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/metric_mapping.csv | header row | PASS | 10 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/metric_mapping.csv | one physical line per record | PASS | 240 | 240 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/metric_mapping.csv | required data rows | PASS | 239 | >0 | required Exp0.1 audit CSV |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/missing_required_fields.csv | pandas.read_csv | PASS | 0 rows / 4 columns | readable CSV | 1 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/missing_required_fields.csv | header row | PASS | 4 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/missing_required_fields.csv | one physical line per record | PASS | 1 | 1 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/official_source_inventory.csv | pandas.read_csv | PASS | 23 rows / 13 columns | readable CSV | 24 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/official_source_inventory.csv | header row | PASS | 13 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/official_source_inventory.csv | one physical line per record | PASS | 24 | 24 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/official_source_inventory.csv | required data rows | PASS | 23 | >0 | required Exp0.1 audit CSV |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/sanity_check_results.csv | pandas.read_csv | PASS | 20 rows / 5 columns | readable CSV | 21 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/sanity_check_results.csv | header row | PASS | 5 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/sanity_check_results.csv | one physical line per record | PASS | 21 | 21 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/sanity_check_results.csv | required data rows | PASS | 20 | >0 | required Exp0.1 audit CSV |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/scenario_mapping.csv | pandas.read_csv | PASS | 36 rows / 9 columns | readable CSV | 37 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/scenario_mapping.csv | header row | PASS | 9 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/scenario_mapping.csv | one physical line per record | PASS | 37 | 37 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/scenario_mapping.csv | required data rows | PASS | 36 | >0 | required Exp0.1 audit CSV |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/schema_profile.csv | pandas.read_csv | PASS | 259 rows / 17 columns | readable CSV | 260 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/schema_profile.csv | header row | PASS | 17 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/schema_profile.csv | one physical line per record | PASS | 260 | 260 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/score_scale_audit.csv | pandas.read_csv | PASS | 5536 rows / 16 columns | readable CSV | 5537 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/score_scale_audit.csv | header row | PASS | 16 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/score_scale_audit.csv | one physical line per record | PASS | 5537 | 5537 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/source_inventory.csv | pandas.read_csv | PASS | 466 rows / 9 columns | readable CSV | 467 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/source_inventory.csv | header row | PASS | 9 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/source_inventory.csv | one physical line per record | PASS | 467 | 467 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_coverage_by_label.csv | pandas.read_csv | PASS | 30 rows / 5 columns | readable CSV | 31 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_coverage_by_label.csv | header row | PASS | 5 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_coverage_by_label.csv | one physical line per record | PASS | 31 | 31 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_coverage_by_metric.csv | pandas.read_csv | PASS | 72 rows / 5 columns | readable CSV | 73 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_coverage_by_metric.csv | header row | PASS | 5 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_coverage_by_metric.csv | one physical line per record | PASS | 73 | 73 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_coverage_by_scenario.csv | pandas.read_csv | PASS | 53 rows / 5 columns | readable CSV | 54 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_coverage_by_scenario.csv | header row | PASS | 5 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_coverage_by_scenario.csv | one physical line per record | PASS | 54 | 54 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_coverage_by_subject.csv | pandas.read_csv | PASS | 140 rows / 5 columns | readable CSV | 141 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_coverage_by_subject.csv | header row | PASS | 5 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_coverage_by_subject.csv | one physical line per record | PASS | 141 | 141 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_label_distribution.csv | pandas.read_csv | PASS | 30 rows / 5 columns | readable CSV | 31 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_label_distribution.csv | header row | PASS | 5 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_label_distribution.csv | one physical line per record | PASS | 31 | 31 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_metric_distribution.csv | pandas.read_csv | PASS | 72 rows / 5 columns | readable CSV | 73 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_metric_distribution.csv | header row | PASS | 5 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_metric_distribution.csv | one physical line per record | PASS | 73 | 73 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_scenario_distribution.csv | pandas.read_csv | PASS | 53 rows / 5 columns | readable CSV | 54 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_scenario_distribution.csv | header row | PASS | 5 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_scenario_distribution.csv | one physical line per record | PASS | 54 | 54 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_stats.csv | pandas.read_csv | PASS | 6 rows / 17 columns | readable CSV | 7 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_stats.csv | header row | PASS | 17 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_stats.csv | one physical line per record | PASS | 7 | 7 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_stats.csv | required data rows | PASS | 6 | >0 | required Exp0.1 audit CSV |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_subject_distribution.csv | pandas.read_csv | PASS | 140 rows / 5 columns | readable CSV | 141 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_subject_distribution.csv | header row | PASS | 5 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/split_subject_distribution.csv | one physical line per record | PASS | 141 | 141 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/subject_mapping.csv | pandas.read_csv | PASS | 51 rows / 8 columns | readable CSV | 52 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/subject_mapping.csv | header row | PASS | 8 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/subject_mapping.csv | one physical line per record | PASS | 52 | 52 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/subject_mapping.csv | required data rows | PASS | 51 | >0 | required Exp0.1 audit CSV |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/unmapped_metrics.csv | pandas.read_csv | PASS | 11 rows / 10 columns | readable CSV | 12 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/unmapped_metrics.csv | header row | PASS | 10 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/unmapped_metrics.csv | one physical line per record | PASS | 12 | 12 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/unmapped_scenarios.csv | pandas.read_csv | PASS | 0 rows / 9 columns | readable CSV | 1 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/unmapped_scenarios.csv | header row | PASS | 9 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/unmapped_scenarios.csv | one physical line per record | PASS | 1 | 1 | header plus one line per parsed data row |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/unmapped_subjects.csv | pandas.read_csv | PASS | 0 rows / 8 columns | readable CSV | 1 physical lines |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/unmapped_subjects.csv | header row | PASS | 8 | >0 |  |
| /Users/sss/edubench-eval/thesis_exp/outputs/exp00_data/tables/unmapped_subjects.csv | one physical line per record | PASS | 1 | 1 | header plus one line per parsed data row |
