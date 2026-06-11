# Exp7-A QD-R1 CORAL Summary

- Dataset: `QD-S0_human_only`, question_seed42.
- Input: fixed A4 text field.
- Head: shared latent quality score plus ordered thresholds.
- Loss: unweighted ordinal BCEWithLogits.
- Synthetic data: no.
- Class weights, focal loss, calibration: no.
- Formal status: `completed`.
- Exp7-B gate: **NOT_RECOMMENDED_YET**.
- Rank consistency solved monotonic violation, but low-score overestimation remains.
