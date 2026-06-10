# Exp6 QD-S2 Human + Synthetic L1 Weighted Ordinal

Status: `completed`
Model: `/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B`
Train: 3326 human + 384 synthetic low-score pseudo-label rows
Dev/test: question_seed42 human-only
Loss: L1 weighted ordinal BCE
Class weights: computed from QD-S2 train only
Checkpoint selection: dev `MAE_label` (min)

## Test Metrics

- Accuracy: 0.6554850407978241
- MAE_label: 0.4566334239951646
- MAE_expected: 0.42833632120932985
- QWK: 0.5466132115577904
- Kendall tau: 0.4716942923251824
- severe_error_rate: 0.04895738893925657
- low_to_high_rate: 0.4838709677419355
- mean_sample_weight: 0.7715906725413557
