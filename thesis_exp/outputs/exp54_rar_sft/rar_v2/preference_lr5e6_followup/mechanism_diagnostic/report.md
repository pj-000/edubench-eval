# SORC-DPO LR=5e-6 train-only mechanism diagnosis

This report uses only frozen train-pair diagnostics. It does not read dev or test and publishes no row-level identifiers or text.

## P2 ordinal-offset effect relative to P1

| Score block | Seed 42 | Seed 43 | Seed 44 | Mean |
|---|---:|---:|---:|---:|
| adjacent_score | -0.000292 | +0.000401 | -0.000583 | -0.000158 |
| severe_l2h | +0.005674 | +0.005674 | +0.005345 | +0.005565 |
| h2l_guard | -0.005181 | -0.003536 | -0.003618 | -0.004112 |

- P2 strict offset satisfaction: 0.00%.
- Severe-L2H P2−P1 beta-margin delta was positive in all seeds.
- H2L-guard P2−P1 beta-margin delta was negative in all seeds.
- P3 rationale contrast-positive rate: 90.15%.

## Interpretation

The ODPO offset was not inert: it consistently redirected train margin toward the severe low-to-high block. However, no score pair reached its full prescribed offset, the high-score guard margin weakened relative to P1, and dev did not establish an independent P2-over-P1 benefit. This is mechanism evidence for risk-conditioned pressure, not confirmatory evidence that the ordinal offset improved generalization.
