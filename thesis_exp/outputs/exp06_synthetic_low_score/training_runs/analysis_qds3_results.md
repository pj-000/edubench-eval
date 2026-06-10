# Exp6 QD-S3 Result Analysis

QD-S3 was completed as a two-stage curriculum run:

- Stage 1: 384 synthetic low-score pseudo-label rows.
- Stage 2: 3326 question_seed42 human-only rows.
- Loss: ordinary ordinal BCEWithLogitsLoss.
- Stage2 checkpoint selection: dev MAE_label minimum.
- Selected stage2 epoch: 6.

## Test Comparison

| run | MAE_label | QWK | Accuracy | low_to_high | Acc@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| QD-B0 | 0.4019 | 0.5976 | 0.6936 | 0.5161 | 0.7873 |
| QD-B1 | 0.4279 | 0.6012 | 0.6709 | 0.4516 | 0.7419 |
| QD-S1 | 0.4418 | 0.5544 | 0.6655 | 0.5484 | 0.7013 |
| QD-S2 | 0.4566 | 0.5466 | 0.6555 | 0.4839 | 0.7029 |
| QD-S3 | 0.4569 | 0.5316 | 0.6247 | 0.6129 | 0.6607 |

## Main Finding

QD-S3 does not improve the Exp6 question-disjoint low-score problem. It is worse than QD-B0, QD-B1, QD-S1, and QD-S2 on the main low_to_high metric.

Compared with QD-B0:

- MAE_label worsens by +0.0550.
- Accuracy drops by -0.0689.
- QWK drops by -0.0660.
- low_to_high worsens by +0.0968.
- Acc@5 drops by -0.1266.

Compared with QD-B1:

- MAE_label worsens by +0.0290.
- QWK drops by -0.0697.
- low_to_high worsens by +0.1613.
- Acc@5 drops by -0.0812.

## Error Pattern

The failure is concentrated in low-score handling, especially label 2:

- Test low_n = 31.
- Test low_to_high = 0.6129, meaning 19 of 31 low-score samples are predicted as high.
- Label 1 Acc@1 is 0.8889, but there are only 9 label-1 test samples.
- Label 2 Acc@2 is 0.0 over 22 label-2 test samples.
- Label 2 mean_pred_label = 3.9545, so almost all label-2 samples are pushed near label 4.

High-score behavior is also damaged:

- Test Acc@5 = 0.6607, versus 0.7873 for QD-B0 and 0.7419 for QD-B1.
- True label-5 underestimation rate is 0.3393.
- The model predicts label 4 for 48.8% of test samples and label 5 for 45.5%, with almost no label 1/2 predictions.

## Stage Dynamics

Stage 1 synthetic-only pretraining does learn a low-score-biased model, but it does not transfer cleanly:

- After stage1, dev predictions are almost entirely label 1/2 and high-score dev performance collapses.
- Stage2 human-only fine-tuning quickly restores high-score behavior.
- By the selected stage2 epoch, the model again predicts almost no label 1/2.

This suggests the synthetic pretraining signal is overwritten during human fine-tuning, or it shifts the representation in a way that does not create a stable label-2 decision boundary.

## Interpretation

The result argues against using the current final384 low-score pool as a simple curriculum pretraining stage with the current ordinary ordinal setup. The synthetic data may still be useful, but not in this QD-S3 configuration.

Likely causes:

- Distribution mismatch between synthetic-only stage1 and human-only stage2.
- The final synthetic pool contains only labels 1/2/3, so stage1 has no high-score calibration.
- Stage2 human training reintroduces the original high-label-dominant bias.
- Label 2 remains the hardest boundary and is not fixed by pretraining.
- Dev selection by overall MAE_label does not directly select for low_to_high reduction.

## Conclusion

QD-S3 should not be reported as successful synthetic augmentation. The strongest Exp6 baseline remains QD-B1 for low_to_high, while QD-B0 remains strongest on MAE_label and Accuracy.

Recommended conclusion wording:

Synthetic low-score data did not improve question-disjoint scoring under simple mixing, L1-weighted mixing, or synthetic-pretrain-then-human-finetune. The main remaining issue is not just synthetic quality, but the interaction between low-score augmentation, label distribution, objective selection, and checkpoint selection.
