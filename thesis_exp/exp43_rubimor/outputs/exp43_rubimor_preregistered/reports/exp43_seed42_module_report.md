# Exp43 Seed42 Module Report

All values are five-fold question-key-disjoint out-of-fold metrics at fixed epoch 10.

| Variant | MAE | QWK | Exact | Kendall | Human RPS | Low-to-high | Label2 recall | Label5 recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.407687 | 0.425271 | 0.664280 | 0.483168 | 0.072914 | 0.750000 | 0.000000 | 0.771180 |
| E1 | 0.391485 | 0.445402 | 0.670686 | 0.507772 | 0.071308 | 0.789474 | 0.000000 | 0.774077 |
| E2 | 0.399397 | 0.430352 | 0.662396 | 0.487460 | 0.056927 | 0.789474 | 0.000000 | 0.766112 |
| E3 | 0.385833 | 0.453104 | 0.674454 | 0.505792 | 0.055689 | 0.750000 | 0.000000 | 0.767560 |
| E4 | 0.380181 | 0.459851 | 0.682743 | 0.512084 | 0.056149 | 0.763158 | 0.000000 | 0.783490 |
| E5 | 0.388470 | 0.425060 | 0.677468 | 0.507615 | 0.056928 | 0.789474 | 0.000000 | 0.774077 |

## Module gates

- Stage 0: `GO`
- Stage 1: `GO`
- Stage 2: `GO`
- Stage 3: `GO`
- Stage 4: `METRIC_HEAD_STOP`
- Stage 5: `NOT_RUN_AFTER_GATE_STOP`
- Stage 6: `NOT_RUN_AFTER_GATE_STOP`
- Stage 8: `NOT_RUN_AFTER_GATE_STOP`

## Interpretation

- E4 passed because it preserved all protection guards and improved MAE over E3 by at least 0.005.
- E5 stopped because QWK protection failed and no preregistered metric-head mechanism improved.
- Stages 5-9 were not authorized after the Stage 4 stop.
- The sealed test set was not parsed or evaluated.
