# Exp12 Monotonic Projection / MAP-OC

Exp12A status: `COMPLETED`
Exp12B status: `NO_COMPLETED_TRAINING_RUNS`

This report uses dev-only checkpoint selection. Test metrics are final evaluation or post-hoc
diagnostic only and are not used for checkpoint selection, tuning, or training decisions.

## Exp12A Decode-Only Projection

- Projected monotonic violation: `0.0000`.
- Raw to projected low-to-high count delta: `0`.
- MAE delta: `0.0009`; QWK delta: `-0.0027`; Acc@5 delta: `0.0000`.
- Low-score test subset is small, so count-level interpretation is necessary.

## Exp12B Train-Time MAP-OC

NO_COMPLETED_TRAINING_RUNS: Exp12B has not produced completed per-epoch metrics yet.

## Method Notes

- Monotonic projection uses exact PAVA onto `q1 >= q2 >= q3 >= q4`, clipped to `[0,1]`.
- The method keeps ordinal threshold semantics and does not sort thresholds.
- Pairwise learning here is supervised ordinal calibration, not a generative policy training method.
- The fixed checkpoint selection policy is `mae_guard_p_gt_3_low_mean` with `delta=0.005` unless a config explicitly changes it.
- All conclusions should center on low-to-high risk under the ordinal scoring constraint.

## Review Checklist

- Verify `uses_test_for_selection` is false.
- Verify projected monotonic violation is zero or near zero when projected rows exist.
- Verify no checkpoints, weights, raw predictions, arrays, or logs are under tracked Exp12 outputs.
