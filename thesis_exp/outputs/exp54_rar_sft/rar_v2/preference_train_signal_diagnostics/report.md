# SORC-DPO train-only signal diagnosis

| Arm | Mean raw margin change | Mean β-scaled change | Relative LoRA update |
|---|---:|---:|---:|
| P1_FIELD_DPO | 0.009596 | 0.000960 | 0.0414% |
| P2_SORC_SCORE | 0.011038 | 0.001104 | 0.0425% |
| P3_JOINT_SORC | 0.004373 | 0.000437 | 0.0415% |

All three P2 score runs had zero examples whose β-scaled policy-reference contrast exceeded the frozen ordinal offset.

Decision: preserve the original formal null result and run only the preregistered LR-only exploratory follow-up (5e-7 → 5e-6), starting with seed 42. No dev/test data were read.
