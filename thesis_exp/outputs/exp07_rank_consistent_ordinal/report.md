# Exp7-A CORAL-style Rank-consistent Ordinal Scorer

Exp7-A fixes the input and data setting, then changes only the ordinal scorer structure.
Exp3 selected A4 as the best input template, so QD-R1 uses the existing A4 text field.
Exp4 showed ordinal scoring is a reasonable objective for the five-point score.
The previous independent ordinal head can produce rank inconsistency such as P(score>3) >
P(score>2).
Exp5 loss ablations did not stably solve low-to-high errors, and Exp6 synthetic augmentation
did not stably exceed human-only baselines. Exp7-A therefore changes the scorer head instead
of generating data or adding another loss penalty.

This is scoring model fine-tuning, not generative SFT.

## Test Metrics

| run | status | MAE_label | QWK | Accuracy | low_to_high | monotonic_violation | Acc@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QD-B0_human_only_ordinary_ordinal | completed | 0.4019 | 0.5976 | 0.6936 | 0.5161 | 0.1985 | 0.7873 |
| QD-B1_human_only_L1_weighted_ordinal | completed | 0.4279 | 0.6012 | 0.6709 | 0.4516 | 0.3119 | 0.7419 |
| QD-R1_CORAL_human_only | completed | 0.4467 | 0.5272 | 0.6555 | 0.5161 | 0.0000 | 0.8977 |

## Required Questions

- Does QD-R1 reduce monotonic_violation_rate? YES
- Does QD-R1 reduce low_to_high_rate? NO
- Does QD-R1 improve MAE_label/QWK/Kendall? NO
- Does QD-R1 hurt Acc@5? NO; Acc@5 increases, but overestimation bias also increases
- Interpretation: rank consistency solved monotonic violation but did not solve low-score overestimation.
- Should Exp7-B start? **NOT_RECOMMENDED_YET**

Model checkpoints live under `thesis_exp/artifacts/` and must not be committed.
