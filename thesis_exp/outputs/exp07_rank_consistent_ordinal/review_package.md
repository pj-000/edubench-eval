# Exp7-A Review Package

Can Exp7-B start? **PENDING_FORMAL_TRAINING**

Scope:

- Implemented only QD-R1 CORAL rank-consistent ordinal scorer.
- No CORN, class-balanced loss, focal loss, calibration, API calls, or synthetic generation.
- Formal training remains pending until the server script is run.

Review focus:

- Confirm QD-S0 train/dev/test are human-only and A4 text is used directly.
- Confirm CORAL logits satisfy z1 >= z2 >= z3 >= z4 and probabilities are monotonic.
- After training, compare QD-R1 against QD-B0 first and QD-B1 second.
