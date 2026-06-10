# QD-S2_human_plus_synthetic_L1

Overall status: **PASS**

Human train plus reviewed low-score synthetic rows for later L1-weighted ordinal training.

Template: `A4_question_answer_metric_rubric_metadata`.

Synthetic labels remain pseudo labels (`label_provenance=pseudo_label`) and are not human labels.

## Splits

| split | rows | human | synthetic | status |
| --- | ---: | ---: | ---: | --- |
| train | 3710 | 3326 | 384 | PASS |
| dev | 1107 | 1107 | 0 | PASS |
| test | 1103 | 1103 | 0 | PASS |

## Leakage

| check | status | details |
| --- | --- | --- |
| dev_human_only | PASS |  |
| dev_row_count | PASS | 1107 |
| test_human_only | PASS |  |
| test_row_count | PASS | 1103 |
| synthetic_rows_only_in_train | PASS | 0 |
| synthetic_label_provenance_pseudo_label | PASS |  |
| human_label_provenance_human_score | PASS |  |
| no_dev_test_question_overlap_for_synthetic_source | PASS | 0 |
| no_dev_test_triple_overlap_for_synthetic_source | PASS | 0 |
| no_duplicate_synthetic_id | PASS | 0 |
| no_duplicate_synthetic_answer_hash | PASS | 0 |
| required_training_fields_present | PASS |  |
| a4_template_fields_present | PASS |  |
| no_rationale_error_type_label_source_in_training_text | PASS | {} |
