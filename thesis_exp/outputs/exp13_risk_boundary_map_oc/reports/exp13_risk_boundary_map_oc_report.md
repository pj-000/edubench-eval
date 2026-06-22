# Exp13 Risk-Boundary MAP-OC

Mode: `scout`
Status: `NO_COMPLETED_RUNS`

Scout ranking is dev-only. Test metrics are not used for checkpoint selection, config selection, or
tuning.

## Dev-Only Selected Rows

No completed Exp13 training rows were found under the local ignored runs directory.

## Method Notes

- Exp13 adds a squared hinge risk-boundary loss on projected `q3 = P(y > 3)` for low-score samples.
- The base setting is Exp12B `train_projection_point_pair` unless a config explicitly switches base.
- PAVA projection preserves ordinal threshold positions; it is not sorting.
- This is a next-stage experiment design for low-to-high risk calibration, not evidence that the problem is solved.
