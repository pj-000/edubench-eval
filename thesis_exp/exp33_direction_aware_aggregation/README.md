# Exp33B Direction-Aware Rubric-Grounded Aggregation

Exp33B builds a CPU-only train-label aggregation and audit layer from the
Exp33A `representative_train` model-reviewed silver posterior. It does not
train, run inference, call an API, use a GPU, or read clean-dev/test data.

The main method is Direction-Aware Rubric-Grounded Aggregation (DRGA):

`p(y | x) proportional prior_human(y)^alpha * prior_global(y)^beta * product_s P_s(obs_s | y)^w_s * direction_penalties(y, x)`

where source confusion matrices and reliability weights are fitted only from
Exp33A representative-train final silver posterior. Five-fold cross-fitting is
stratified by final silver point label and language; each held-out fold is
predicted using only the other folds.

Public outputs are aggregate metrics, reports, decisions, and hashes. Row-level
cross-fit predictions and full 2,654-row train supervision are written only
under `outputs/**/private/`, which is gitignored by this directory.

Run:

```bash
bash thesis_exp/scripts/run_exp33b_direction_aware_aggregation.sh
```
