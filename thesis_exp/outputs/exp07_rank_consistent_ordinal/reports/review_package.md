# Exp7-A Review Package

Can Exp7-B start? **NOT_RECOMMENDED_YET**

Scope:

- Implemented only QD-R1 CORAL rank-consistent ordinal scorer.
- No CORN, class-balanced loss, focal loss, calibration, API calls, or synthetic generation.
- Formal training status: completed.
- Rank consistency is solved, but low-score overestimation is not solved.

Review focus:

- Confirm QD-S0 train/dev/test are human-only and A4 text is used directly.
- Confirm CORAL logits satisfy z1 >= z2 >= z3 >= z4 and probabilities are monotonic.
- Use `reports/qdr1_diagnosis.md` before deciding whether any Exp7-B variant is justified.
