# Exp6-12 Final Synthetic Low-score Pool Report

## Summary

- Final pool count: **384**
- Final label distribution: `{'1': 168, '2': 168, '3': 48}`
- Language distribution: `{'en': 194, 'zh': 190}`
- Metric coverage: **12**
- Error type coverage: **7**
- Accepted count: **374**
- Revised label count: **0**
- Revised error_type count: **10**
- Rejected count: **0**
- Replacement count: **0**

## Curation Policy

Exp6-12 used conservative local rule-based quality checks only. It did not call APIs, generate new synthetic answers, or train a model. Synthetic labels remain `synthetic_design` / `pseudo_label`, not human labels.

## Gates

- Can Exp6 training dataset build start: **YES**
- Can Exp6 model training start: **NO** until training datasets are built and reviewed
- Remaining blockers: **None for training dataset build; model training remains blocked until dataset build/review is complete.**

## Sanity Checks

| check_name | status | count | notes |
| --- | --- | ---: | --- |
| final_count_384 | PASS | 384 | final synthetic low-score pool |
| label_distribution_168_168_48 | PASS | 384 | {"1": 168, "2": 168, "3": 48} |
| language_distribution_reported | INFO | 384 | {"en": 194, "zh": 190} |
| metric_coverage_12 | PASS | 12 | 12 metrics expected |
| error_type_coverage_7 | PASS | 7 | 7 error types expected |
| no_duplicate_synthetic_id | PASS | 0 | duplicate synthetic_id count |
| no_duplicate_answer_hash | PASS | 0 | normalized answer hash duplicate count |
| all_source_split_train | PASS | 0 | source_split must be train |
| no_dev_test_source_question_overlap | PASS | 0 | source_question_key overlap with dev/test |
| no_dev_test_source_triple_overlap | PASS | 0 | source_triple_key overlap with dev/test |
| no_api_key_in_final_curated_files | PASS | 0 | secret-like marker scan |
| no_checkpoint_or_weights_in_final_curated | PASS | 0 |  |
| label_source_synthetic_design | PASS | 0 | label_source must remain synthetic_design |
| label_provenance_pseudo_label | PASS | 0 | label_provenance must remain pseudo_label |
| api_not_called | PASS | 0 | Exp6-12 used local files only |
| training_not_run | PASS | 0 | No training command executed |
