# Exp16A Boundary Linking Results

This report summarizes lightweight synced results from `~/edubench-eval-exp2` on the server. Checkpoints, raw predictions, arrays, and full run artifacts are not included.

## Run Setup

- Variants: `global`, `metric_rubric`, `qmr`, `qmr_meta`.
- Seed: `42`.
- Data split: `question_seed42`.
- Default selection: best checkpoint by dev MAE.
- Test metrics are final held-out evaluation only, not used for variant or checkpoint selection.

## Dev Ranking

| Rank | Variant | MAE | QWK | Accuracy | Low-to-high | Monotonic violation | Selected epoch |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `qmr` | 0.3993 | 0.5532 | 0.6766 | 29/57 (0.5088) | 0.0000 | 3 |
| 2 | `qmr_meta` | 0.4092 | 0.5046 | 0.6694 | 30/57 (0.5263) | 0.0000 | 3 |
| 3 | `metric_rubric` | 0.4237 | 0.5274 | 0.6531 | 33/57 (0.5789) | 0.0000 | 3 |
| 4 | `global` | 0.6007 | 0.3500 | 0.5393 | 37/57 (0.6491) | 0.0000 | 3 |

## Test Metrics

| Variant | MAE | QWK | Accuracy | Low-to-high | High-to-low | Monotonic violation |
|---|---:|---:|---:|---:|---:|---:|
| `global` | 0.5177 | 0.3046 | 0.5893 | 20/31 (0.6452) | 1/967 (0.0010) | 0.0000 |
| `metric_rubric` | 0.3708 | 0.5524 | 0.6736 | 15/31 (0.4839) | 0/967 (0.0000) | 0.0000 |
| `qmr` | 0.3772 | 0.5380 | 0.6718 | 17/31 (0.5484) | 0/967 (0.0000) | 0.0000 |
| `qmr_meta` | 0.3953 | 0.4964 | 0.6491 | 17/31 (0.5484) | 0/967 (0.0000) | 0.0000 |

## Quick Read

- Best dev MAE variant: `qmr` with dev MAE `0.3993` and QWK `0.5532`.
- Lowest test MAE variant: `metric_rubric` with test MAE `0.3708` and QWK `0.5524`.
- All variants report monotonic violation rate `0.0000`, matching the ordered-threshold design.
- Low-to-high remains high in this scout; Exp16A should be read primarily as a boundary-structure diagnostic, not as a solved risk-reduction method.
