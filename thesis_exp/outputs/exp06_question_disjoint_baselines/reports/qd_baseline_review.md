# Exp6 Question-disjoint Baseline Review

Overall status: **PASS**

Scope: QD-B0 and QD-B1 rerun the human-only baselines on `question_seed42` so Exp6
synthetic-generation results have a question-disjoint comparison point.

## Dataset

| split | rows | expected_rows | status | path |
| --- | --- | --- | --- | --- |
| train | 3326 | 3326 | PASS | thesis_exp/outputs/exp06_question_disjoint_baselines/datasets/A4_ques... |
| dev | 1107 | 1107 | PASS | thesis_exp/outputs/exp06_question_disjoint_baselines/datasets/A4_ques... |
| test | 1103 | 1103 | PASS | thesis_exp/outputs/exp06_question_disjoint_baselines/datasets/A4_ques... |

Dataset card:
`thesis_exp/outputs/exp06_question_disjoint_baselines/datasets/A4_question_answer_metric_rubric_metadata_question_seed42/dataset_card.md`

## Runs

| run_id | status | split | n | MAE_label | MAE_expected | Exact Match | Macro-F1 | Quadratic Weighted Kappa | low_to_high_rate | epoch | global_step | checkpoint_state_dict_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QD-B0_human_only_ordinary_ordinal | completed | dev | 1107 | 0.4658235471243601 | 0.4202975068149761 | 0.6766034327009937 | 0.5032087688277459 | 0.5650525008529019 | 0.5263157894736842 | best | 260 |  |
| QD-B0_human_only_ordinary_ordinal | completed | test | 1103 | 0.4019341190692051 | 0.37631541153116166 | 0.6935630099728014 | 0.5465464782390519 | 0.5976217823018422 | 0.5161290322580645 | best | 260 |  |
| QD-B1_human_only_L1_weighted_ordinal | completed | dev | 1107 | 0.48118036735922903 | 0.4304336130845666 | 0.6485998193315267 | 0.5346131073141539 | 0.5614556962025317 | 0.49122807017543857 | best | 260 |  |
| QD-B1_human_only_L1_weighted_ordinal | completed | test | 1103 | 0.42792384406165007 | 0.3980361290190023 | 0.670897552130553 | 0.5324847212262678 | 0.6012356007375432 | 0.45161290322580644 | best | 260 |  |

## Test Comparison

| metric | QD-B0_test | QD-B1_test | B1_minus_B0 |
| --- | --- | --- | --- |
| MAE_label | 0.401934 | 0.427924 | 0.0259897 |
| MAE_expected | 0.376315 | 0.398036 | 0.0217207 |
| Exact Match | 0.693563 | 0.670898 | -0.0226655 |
| Macro-F1 | 0.546546 | 0.532485 | -0.0140618 |
| Quadratic Weighted Kappa | 0.597622 | 0.601236 | 0.00361382 |
| low_to_high_rate | 0.516129 | 0.451613 | -0.0645161 |

## Artifact Guardrails

- Checkpoint directory: `thesis_exp/artifacts/exp06_question_disjoint_baselines/checkpoints`
- Tracked checkpoint files: **0**
