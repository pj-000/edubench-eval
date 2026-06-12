# Exp8 QD-ER1 EduRisk Run Summary

Status: `completed`
Run ID: `QD-ER1_EduRisk_human_only`
Model: `/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B`
Dataset: `QD-S0_human_only`, question_seed42 human-only train/dev/test.
Fixed input: `A4_question_answer_metric_rubric_metadata` text field.
Head: CORAL-style rank-consistent cumulative ordinal head.
Loss: soft ordinal CE plus normalized low-score risk plus cumulative BCE.
Synthetic rows: none.
Class-balanced effective-number beta: `0.99`.
Checkpoint selection: dev `MAE_label` (min)

## Test Metrics

- MAE_label: 0.4378966455122393
- QWK: 0.5580654633188158
- Acc@5: 0.8035714285714286
- low_to_high_rate: 0.45161290322580644
- monotonic_violation_rate: 0.0
- expected_edurisk: 0.24929827409382682
