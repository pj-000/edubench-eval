# Exp49-CPHCE seed42 analysis

Final status: **SEED42_NO_GO**

The seed42 gate failed under the locked paper-compatible checkpoint rule. Seeds
43/44 were not launched, no manifest was frozen, and Exp49 did not access test.

## Selected checkpoints

| Arm | Epoch | Exact | MAE (human mean) | Bias | Kendall | L2H | QWK | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 hard CE | 8 | 0.718373 | 0.386044 | +0.113956 | 0.577077 | 11/20 | 0.623193 | 0.147713 |
| M1 human-soft CE | 5 | 0.713855 | 0.389056 | +0.101908 | 0.576397 | 8/20 | 0.648453 | 0.049738 |
| M1 - B0 | — | -0.004518 | +0.003012 | -0.012048 | -0.000680 | -3 | +0.025260 | -0.097975 |

M1 improved absolute bias, low-to-high errors, QWK, NLL, Brier, ECE, expected
score MAE, label-3 recall, and label-5 recall. It nevertheless failed both
primary seed42 gates: MAE improvement was `-0.003012` instead of at least
`+0.005`, and Exact fell by 3/664 instead of at most 2/664. The paired row
bootstrap for `M1 MAE - B0 MAE` was `0.003012 [−0.019578, 0.026104]`.

## Every-epoch dev metrics

| Epoch | B0 Exact | M1 Exact | B0 MAE | M1 MAE | B0 Kendall | M1 Kendall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.590361 | 0.626506 | 0.530622 | 0.506526 | 0.292706 | 0.349382 |
| 2 | 0.643072 | 0.667169 | 0.500502 | 0.439759 | 0.412802 | 0.499703 |
| 3 | 0.656627 | 0.688253 | 0.476406 | 0.418675 | 0.485548 | 0.541061 |
| 4 | 0.695783 | 0.695783 | 0.419679 | 0.396586 | 0.537669 | 0.557719 |
| 5 | 0.716867 | 0.713855 | 0.396586 | 0.389056 | 0.589379 | 0.576397 |
| 6 | 0.707831 | 0.697289 | 0.393574 | 0.395582 | 0.577510 | 0.573780 |
| 7 | 0.712349 | 0.704819 | 0.392570 | 0.389056 | 0.580106 | 0.581974 |
| 8 | 0.718373 | 0.706325 | 0.386044 | 0.383032 | 0.577077 | 0.592899 |
| 9 | 0.712349 | 0.704819 | 0.391064 | 0.384538 | 0.575279 | 0.588650 |
| 10 | 0.716867 | 0.710843 | 0.388052 | 0.384036 | 0.575600 | 0.587161 |

The early-epoch improvement is real but is not the registered result: each arm
must use its own highest-Exact checkpoint. B0's epoch 8 therefore competes with
M1's epoch 5. Changing selection after seeing this table would be post-hoc
metric drift.

## Paired error analysis

- M1 fixed 45 B0 Exact errors and broke 48 B0 correct predictions: net `-3`.
- On 373 two-of-three-majority rows, M1 improved discrete MAE by `0.018767`
  and expected-score MAE by `0.086873` on average.
- On 291 unanimous rows, M1 worsened those values by `0.030928` and `0.063781`.
- Label-4 recall fell from 164/237 to 152/237; label-5 recall rose from
  289/345 to 291/345.
- The input text hashes were identical, and neither split had truncation.

The evidence supports a narrower hypothesis: annotator disagreement is useful,
but the full empirical distribution is too strong for a system that is selected
and judged primarily by rounded-label Exact. Any follow-up must be registered as
a new experiment; Exp49 itself is closed and must not be reinterpreted.

