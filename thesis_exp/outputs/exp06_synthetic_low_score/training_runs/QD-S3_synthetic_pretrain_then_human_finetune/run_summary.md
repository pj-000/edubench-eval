# Exp6 QD-S3 Synthetic Pretrain then Human Fine-tune

Status: `completed`
Model: `/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B`
Stage 1 train: 384 synthetic low-score pseudo-label rows
Stage 2 train: 3326 question_seed42 human-only rows
Dev/test: question_seed42 human-only
Loss: ordinary ordinal BCEWithLogitsLoss
Class weights: disabled
Asymmetric loss: disabled
Stage 1 epochs: 3.0
Stage 2 epochs: 10.0
Checkpoint selection: dev `MAE_label` (min)
Selected stage2 epoch: 6

## Test Metrics

- Accuracy: 0.6246600181323663
- MAE_label: 0.45693563009972804
- MAE_expected: 0.42699128785385876
- QWK: 0.5315848261086693
- Kendall tau: 0.4574926422163617
- severe_error_rate: 0.0371713508612874
- low_to_high_rate: 0.6129032258064516
- monotonic_violation_rate: 0.314596554850408
