# Exp33B Direction-Aware Rubric-Grounded Aggregation

- Reference source: Exp33A `representative_train` model-reviewed silver posterior.
- Calibration rows: 120; cross-fitting: 5 folds stratified by silver point label and language.
- Paper protocol preserved: train/dev/test remain the locked 2654/664/2218 triple-key-isolated split.
- Dev/test access: none.
- Training/API/GPU/student inference: none.
- Risk view stress test: COMPLETE; risk view is not used as prevalence.
- Full train supervision generated: 0 rows.

## DRGA formula

`p(y|x) proportional H_x(y)^alpha * pi(y)^beta * product_s C_s[y, obs_s]^w_s * D_y(x)`

`H_x` is the human three-score empirical distribution, `pi` is the fold training
silver prior, `C_s` is the ordinal source confusion matrix fitted from
model-reviewed silver posterior, and `D_y` applies preregistered direction and
score-cap penalties. High-entropy or low-confidence rows fall back exactly to
the human empirical distribution.

## Cross-fit headline

| method | rows | MAE | QWK | low_to_high | high_to_low | fallback_rate |
|---|---:|---:|---:|---:|---:|---:|
| rounded_human | 120 | 0.4516185213095537 | 0.5695764379025433 | 0.011881436824918363 | 0.009126685087498953 | 0.0 |
| DRGA | 120 | 0.35387578993846225 | 0.703674173625964 | 0.0 | 0.011821150464707358 | 0.0 |

Best mature aggregation baseline by MAE: `qwen` with MAE
`0.2687924785588689` and QWK `0.8077023911033611`. DRGA is not
declared superior unless the table supports that claim.

## Gate decision

- Passed: False.
- Gates: `{"drga_not_worse_high_to_low": false, "drga_not_worse_low_to_high": true, "drga_not_worse_than_rounded_human_mae": true, "drga_not_worse_than_rounded_human_qwk": true, "fallback_recomputable": true, "no_dev_or_test_access": true, "posterior_valid": true}`.

This is a model-reviewed silver aggregation experiment, not human expert gold.
