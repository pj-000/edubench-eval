# Exp59 reproducibility package

This directory contains the public, CPU-recomputable evidence for the frozen
five-seed Exp59 development-set confirmation.

## Included endpoints

- Reused `consensus_only` endpoints from Exp57 for seeds 42--46.
- Exp59 `parallel_only` and `orthogonal_only` endpoints for seeds 42--46.
- For every Exp59 endpoint: the run summary, full development history,
  selected metrics, selected development predictions, geometry-step audit and
  first-64-step training trace.
- For every reused Exp57 endpoint: the already tracked run summary, full
  development history and selected metrics, plus the selected development
  predictions needed by the cluster bootstrap.

The prediction rows include hashed `question_key`, `record_id` and
`triple_key` fields. They do not expose response text.

## Recompute

From the repository root, run:

```bash
python -m thesis_exp.exp59_residual_geometry.analyze_confirmation
```

The command regenerates:

- `audit/exp59_five_seed_confirmation.json`
- `tables/exp59_five_seed_comparison.csv`
- `audit/exp59_checkpoint_sensitivity.json`
- `tables/exp59_checkpoint_sensitivity.csv`

The bootstrap estimates the item-weighted mean development-set MAE difference
with question-cluster resampling, conditional on the five trained seeds. The
primary comparison, thresholds, seeds and cluster-CI rule were frozen before
the results. The exact analysis implementation, 10,000 repetitions and random
seed 57 were implemented after the results and are not described as fully
preregistered code.

## Scope

This is an internal development-set mechanism confirmation under the frozen
standard-clipping and AdamW setting. It supports a stable non-collinear
raw-gradient contribution. It does not establish a pure orthogonal causal
effect, function-space orthogonality or cross-dataset generalization.
