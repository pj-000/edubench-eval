# Exp9 QD-PR1 Pairwise Result Summary

Formal status: `completed`.

QD-PR1 did not reduce low_to_high: `0.5161` vs QD-B1 `0.4516`.

QD-PR1 did not beat QD-B1: MAE `0.4545` vs `0.4279`, QWK `0.5473` vs `0.6012`, Acc@5 `0.6916` vs
`0.7419`.

Main failure: pairwise training damaged ordinal monotonicity (`0.7616`) and therefore worsened
pointwise calibration, despite useful dev pairwise ranking signal.

Interpretation: negative QD-PR1 result; pairwise direction remains promising only as anchored
fine-tuning.

QD-PR2 recommendation: initialize from QD-B1, use `lambda_pair` in `{0.05, 0.1}`, add monotonic
regularization, use high-comparability pairs only, and fine-tune for 2-3 epochs.
