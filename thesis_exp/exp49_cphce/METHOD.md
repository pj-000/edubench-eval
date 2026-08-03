# Method contract

For human ratings `r1`, `r2`, and `r3`, the baseline target is the existing
rounded label `y`. The treatment target is

`d[k] = count(rj == k) / 3`.

The losses are:

- Hard: `-log p[y]`
- CPHCE: `-sum_k d[k] log p[k]`

No hard-loss mixture, weighting, auxiliary head, teacher, calibration,
resampling, or synthetic data is allowed. Formal predictions are raw-logit
argmax classes. Exact uses the rounded label; MAE, signed bias, and Kendall use
the continuous three-rater mean.

The paper Bin Agreement has been recovered as low/mid/high agreement with
`{1,2}`, `{3}`, and `{4,5}`. It reproduces all five PDF evaluator values within
0.003 on the slightly different reconstructed local split.

Exp49 deliberately keeps the original class imbalance. Per-label recall and
low-to-high counts are mandatory diagnostics, but the 20-row dev low tail is not
a single-seed veto. Formal conclusions require paired three-seed evidence.
