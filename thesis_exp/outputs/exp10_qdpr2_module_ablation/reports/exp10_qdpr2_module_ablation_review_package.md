# Exp10 QD-PR2 Module Ablation Review Package

## What Changed

- Formal matrix now contains seven runs, including `no_pair_same_pair_batches`.
- `no_pair` is retained as `no_pair_pointwise_loader` auxiliary diagnostic evidence.
- `no_pair_same_pair_batches` is the strict L_pair ablation: pair loader is used, `force_pair_training=true`, and `lambda_pair=0`.

## Expected Run Order

1. `full_qdpr2`: loader `pair`, active losses `L_point + L_pair + L_anchor + L_mono`, strict `False`
2. `no_pair`: loader `pointwise`, active losses `L_point + L_anchor + L_mono`, strict `False`
3. `no_pair_same_pair_batches`: loader `pair`, active losses `L_point + L_anchor + L_mono`, strict `True`
4. `no_anchor`: loader `pair`, active losses `L_point + L_pair + L_mono`, strict `True`
5. `no_mono`: loader `pair`, active losses `L_point + L_pair + L_anchor`, strict `True`
6. `point_only`: loader `pointwise`, active losses `L_point`, strict `False`
7. `no_point_diagnostic`: loader `pair`, active losses `L_pair + L_anchor + L_mono`, strict `False`

## Interpretation Notes

The strict run is the main L_pair evidence under identical pair-batch exposure. The older `no_pair`
run is useful for diagnosing how the pointwise-loader path behaves, but it should not be treated as
the strict module comparison.

Use cautious wording in the thesis discussion: the table can suggest module effects, while the
low-score test subset is small and count-level interpretation is necessary.

## Output Tables

- `thesis_exp/outputs/exp10_qdpr2_module_ablation/tables/exp10_ablation_metrics_summary.csv`
- `thesis_exp/outputs/exp10_qdpr2_module_ablation/tables/exp10_ablation_module_effects.csv`
- `thesis_exp/outputs/exp10_qdpr2_module_ablation/tables/exp10_ablation_low_score_distribution.csv`
- `thesis_exp/outputs/exp10_qdpr2_module_ablation/tables/exp10_ablation_low_to_high_by_label.csv`
- `thesis_exp/outputs/exp10_qdpr2_module_ablation/tables/exp10_ablation_monotonic_by_threshold.csv`
- `thesis_exp/outputs/exp10_qdpr2_module_ablation/tables/exp10_ablation_vs_existing_baselines.csv`

## Current Completion

| ablation | status | loader | force pair | strict |
| --- | --- | --- | ---: | ---: |
| full_qdpr2 | missing | pair | False | False |
| no_pair | missing | pointwise | False | False |
| no_pair_same_pair_batches | missing | pair | True | True |
| no_anchor | missing | pair | False | True |
| no_mono | missing | pair | False | True |
| point_only | missing | pointwise | False | False |
| no_point_diagnostic | missing | pair | False | False |
