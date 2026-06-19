# Exp10 QD-PR2 Module Ablation Report

Completed runs: `0` / `7`
Pending runs: `full_qdpr2, no_pair, no_pair_same_pair_batches, no_anchor, no_mono, point_only,
no_point_diagnostic`

This collector uses question-disjoint test predictions only. It does not train models and does not
use test labels for model selection.

## Ablation Matrix

| ablation | display | loader | force pair | strict | active losses | low-to-high | MAE | QWK |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| full_qdpr2 | full_qdpr2 | pair | False | False | L_point + L_pair + L_anchor + L_mono | pending | pending | pending |
| no_pair | no_pair_pointwise_loader | pointwise | False | False | L_point + L_anchor + L_mono | pending | pending | pending |
| no_pair_same_pair_batches | no_pair_same_pair_batches | pair | True | True | L_point + L_anchor + L_mono | pending | pending | pending |
| no_anchor | no_anchor | pair | False | True | L_point + L_pair + L_mono | pending | pending | pending |
| no_mono | no_mono | pair | False | True | L_point + L_pair + L_anchor | pending | pending | pending |
| point_only | point_only | pointwise | False | False | L_point | pending | pending | pending |
| no_point_diagnostic | no_point_diagnostic | pair | False | False | L_pair + L_anchor + L_mono | pending | pending | pending |

## Strict L_pair Ablation with Same Pair Batches

The strict run `no_pair_same_pair_batches` is the main evidence for the L_pair module because it
keeps pair-batch exposure while setting `lambda_pair=0`. The older `no_pair` run is auxiliary
diagnostic evidence because it removes L_pair and uses the pointwise loader.

Interpretation should stay cautious: results can suggest effects under identical pair-batch
exposure, and the low-score test subset is small, so count-level interpretation is necessary.

| run | loader | force_pair_training | lambda_pair | low-to-high | MAE |
| --- | --- | ---: | ---: | --- | ---: |
| no_pair | pointwise | False | 0.0 | pending | pending |
| no_pair_same_pair_batches | pair | True | 0.0 | pending | pending |

## Required Tables

- `thesis_exp/outputs/exp10_qdpr2_module_ablation/tables/exp10_ablation_metrics_summary.csv`
- `thesis_exp/outputs/exp10_qdpr2_module_ablation/tables/exp10_ablation_module_effects.csv`
- `thesis_exp/outputs/exp10_qdpr2_module_ablation/tables/exp10_ablation_low_score_distribution.csv`
- `thesis_exp/outputs/exp10_qdpr2_module_ablation/tables/exp10_ablation_low_to_high_by_label.csv`
- `thesis_exp/outputs/exp10_qdpr2_module_ablation/tables/exp10_ablation_monotonic_by_threshold.csv`
- `thesis_exp/outputs/exp10_qdpr2_module_ablation/tables/exp10_ablation_vs_existing_baselines.csv`

## Module Effects

The module-effects table distinguishes `no_pair` from `no_pair_same_pair_batches` through
`dataloader_mode`, `force_pair_training`, and the interpretation column.
