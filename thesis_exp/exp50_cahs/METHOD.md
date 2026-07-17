# Locked method

For hard consensus `m`, three-rater distribution `d`, and fixed `alpha=0.5`:

`t = 0.5 m + 0.5 d`.

This is exactly equivalent to an equal-weight average of hard CE and human-soft
CE because cross-entropy is linear in its target. Unanimous rows remain one-hot.
Every disputed train/dev row in the fixed data is a 2:1 split between adjacent
scores, so a disputed target contains `5/6` mass on the rounded consensus and
`1/6` on the adjacent minority score.

The method tests only `alpha=0.5`. Failure does not imply every internal alpha
fails, and no additional alpha may be tried under Exp50.
