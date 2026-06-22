# Exp12 Monotonic Projection / MAP-OC

Exp12A status: `COMPLETED`
Exp12B status: `COMPLETED`

This report uses dev-only checkpoint selection. Test metrics are final evaluation or post-hoc
diagnostic only and are not used for checkpoint selection, tuning, or training decisions.

## Exp12A Decode-Only Projection

- Projected monotonic violation: `0.0000`.
- Raw to projected low-to-high count delta: `0`.
- MAE delta: `0.0009`; QWK delta: `-0.0027`; Acc@5 delta: `0.0000`.
- Low-score test subset is small, so count-level interpretation is necessary.

## Exp12B Train-Time MAP-OC

| run | seed | selected epoch | test low-to-high | count | MAE | QWK | monotonic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| map_oc_full | 42 | 3 | 0.4516 | 14 | 0.4204 | 0.5988 | 0.0000 |
| train_projection_point_pair | 42 | 3 | 0.3871 | 12 | 0.4158 | 0.6108 | 0.0000 |
| train_projection_score | 42 | 3 | 0.4194 | 13 | 0.4176 | 0.6063 | 0.0000 |

If MAP-OC lowers monotonic violation but worsens low-to-high, this should be treated as a negative
result. If it lowers low-to-high with MAE/QWK tradeoff, report both sides.

## Method Notes

- Monotonic projection uses exact PAVA onto `q1 >= q2 >= q3 >= q4`, clipped to `[0,1]`.
- The method keeps ordinal threshold semantics and does not sort thresholds.
- Pairwise learning here is supervised ordinal calibration, not a generative policy training method.
- The fixed checkpoint selection policy is `mae_guard_p_gt_3_low_mean` with `delta=0.005` unless a config explicitly changes it.
- All conclusions should center on low-to-high risk under the ordinal scoring constraint.
