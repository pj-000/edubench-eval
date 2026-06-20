# Exp10 QD-PR2 Module Ablation Report

Completed runs: `7` / `7`
Pending runs: `none`

This collector uses question-disjoint test predictions only. It does not train models and does not
use test labels for model selection.

## Ablation Matrix

| ablation | display | loader | force pair | strict | active losses | low-to-high | MAE | QWK |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| full_qdpr2 | full_qdpr2 | pair | False | False | L_point + L_pair + L_anchor + L_mono | 14 / 31 = 0.4516 | 0.4146 | 0.6028 |
| no_pair | no_pair_pointwise_loader | pointwise | False | False | L_point + L_anchor + L_mono | 14 / 31 = 0.4516 | 0.4195 | 0.6041 |
| no_pair_same_pair_batches | no_pair_same_pair_batches | pair | True | True | L_point + L_anchor + L_mono | 14 / 31 = 0.4516 | 0.4210 | 0.6006 |
| no_anchor | no_anchor | pair | False | True | L_point + L_pair + L_mono | 15 / 31 = 0.4839 | 0.4243 | 0.5861 |
| no_mono | no_mono | pair | False | True | L_point + L_pair + L_anchor | 13 / 31 = 0.4194 | 0.4219 | 0.6025 |
| point_only | point_only | pointwise | False | False | L_point | 13 / 31 = 0.4194 | 0.4149 | 0.6089 |
| no_point_diagnostic | no_point_diagnostic | pair | False | False | L_pair + L_anchor + L_mono | 14 / 31 = 0.4516 | 0.4291 | 0.5995 |

## Strict L_pair Ablation with Same Pair Batches

The strict run `no_pair_same_pair_batches` is the main evidence for the L_pair module because it
keeps pair-batch exposure while setting `lambda_pair=0`. The older `no_pair` run is auxiliary
diagnostic evidence because it removes L_pair and uses the pointwise loader.

Interpretation should stay cautious: results can suggest effects under identical pair-batch
exposure, and the low-score test subset is small, so count-level interpretation is necessary.

| run | loader | force_pair_training | lambda_pair | low-to-high | MAE |
| --- | --- | ---: | ---: | --- | ---: |
| no_pair | pointwise | False | 0.0 | 14 / 31 = 0.4516 | 0.4195 |
| no_pair_same_pair_batches | pair | True | 0.0 | 14 / 31 = 0.4516 | 0.4210 |

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

| ablation | metric | delta vs full | interpretation |
| --- | --- | ---: | --- |
| full_qdpr2 | low_to_high_rate | 0.0000 | Reference QD-PR2 objective with all planned loss terms active. |
| no_pair | low_to_high_rate | 0.0000 | Auxiliary diagnostic: removes L_pair and switches to the pointwise loader. |
| no_pair_same_pair_batches | low_to_high_rate | 0.0000 | Primary L_pair evidence: removes the L_pair gradient contribution under identical pair-batch exposure. |
| no_anchor | low_to_high_rate | 0.0323 | Strict module ablation for L_anchor while retaining pair-batch exposure. |
| no_mono | low_to_high_rate | -0.0323 | Strict module ablation for L_mono while retaining pair-batch exposure. |
| point_only | low_to_high_rate | -0.0323 | Reduced pointwise-only baseline with pair, anchor, and monotonic terms disabled. |
| no_point_diagnostic | low_to_high_rate | 0.0000 | Diagnostic run without direct pointwise supervision. |
