# Exp3 Input Ablation Datasets

Derived from the locked Exp0.1 paper-like triple split. Exp3 changes only input templates;
the model family, 5-class CE objective, labels, train/dev/test split, and checkpoint
selection policy remain aligned with Exp2.

A2 is the Exp2-compatible question + answer + metric baseline and should normally reuse
Exp2 formal outputs instead of retraining.

Rubric mode used for generated rows: **corrected**.

## Dataset Stats

| template | split | rows | expected | status |
| --- | --- | ---: | ---: | --- |
| A0_answer_only | train | 2654 | 2654 | PASS |
| A0_answer_only | dev | 664 | 664 | PASS |
| A0_answer_only | test | 2218 | 2218 | PASS |
| A1_question_answer | train | 2654 | 2654 | PASS |
| A1_question_answer | dev | 664 | 664 | PASS |
| A1_question_answer | test | 2218 | 2218 | PASS |
| A2_question_answer_metric | train | 2654 | 2654 | PASS |
| A2_question_answer_metric | dev | 664 | 664 | PASS |
| A2_question_answer_metric | test | 2218 | 2218 | PASS |
| A3_question_answer_metric_rubric | train | 2654 | 2654 | PASS |
| A3_question_answer_metric_rubric | dev | 664 | 664 | PASS |
| A3_question_answer_metric_rubric | test | 2218 | 2218 | PASS |
| A4_question_answer_metric_rubric_metadata | train | 2654 | 2654 | PASS |
| A4_question_answer_metric_rubric_metadata | dev | 664 | 664 | PASS |
| A4_question_answer_metric_rubric_metadata | test | 2218 | 2218 | PASS |

## Token Length Summary

| template | split | mean_token_length | p95_token_length | truncation_rate |
| --- | --- | ---: | ---: | ---: |
| A0_answer_only | train | 276.15 | 764.00 | 0.000000 |
| A0_answer_only | dev | 282.52 | 806.40 | 0.000000 |
| A0_answer_only | test | 286.18 | 827.00 | 0.000000 |
| A1_question_answer | train | 355.09 | 863.00 | 0.000000 |
| A1_question_answer | dev | 360.83 | 877.85 | 0.000000 |
| A1_question_answer | test | 355.28 | 922.00 | 0.000000 |
| A2_question_answer_metric | train | 369.38 | 878.00 | 0.000000 |
| A2_question_answer_metric | dev | 375.11 | 893.85 | 0.000000 |
| A2_question_answer_metric | test | 369.37 | 936.30 | 0.000000 |
| A3_question_answer_metric_rubric | train | 477.33 | 1043.00 | 0.000000 |
| A3_question_answer_metric_rubric | dev | 482.34 | 1051.55 | 0.000000 |
| A3_question_answer_metric_rubric | test | 477.22 | 1102.00 | 0.000000 |
| A4_question_answer_metric_rubric_metadata | train | 501.73 | 1069.00 | 0.000000 |
| A4_question_answer_metric_rubric_metadata | dev | 506.71 | 1076.55 | 0.000000 |
| A4_question_answer_metric_rubric_metadata | test | 501.73 | 1127.30 | 0.000000 |
