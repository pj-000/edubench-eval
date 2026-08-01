# Exp56 MeanAux three-seed development-only report

Status: **EXP56_THREE_SEED_DEV_ONLY_COMPLETED**

| Arm | MAE | Exact | Kendall |
|---|---:|---:|---:|
| Hard-only | 0.393 ± 0.016 | 0.719 ± 0.011 | 0.569 ± 0.024 |
| HMSA | 0.379 ± 0.003 | 0.731 ± 0.005 | 0.597 ± 0.007 |
| MeanAux | 0.395 ± 0.009 | 0.710 ± 0.003 | 0.569 ± 0.006 |

MeanAux does not improve the matched Hard-only baseline on average, while HMSA outperforms MeanAux
on MAE, Exact Match, and Kendall tau for every seed. Because continuous mean and empirical
distribution are one-to-one in these splits, this supports a target-geometry/loss explanation, not
an additional-information claim.

No test data were accessed.
