# Exp10 QD-PR2 Module Ablation Summary

Completed runs: `0` / `6`
Pending runs: `full_qdpr2, no_pair, no_anchor, no_mono, point_only, no_point_diagnostic`

This collector uses question-disjoint test predictions only. It does not train models and does not
use test labels for model selection.

| ablation | MAE | QWK | Accuracy | Acc@5 | low-to-high | monotonic | p1<p2 | p2<p3 | p3<p4 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| full_qdpr2 | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| no_pair | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| no_anchor | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| no_mono | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| point_only | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| no_point_diagnostic | pending | pending | pending | pending | pending | pending | pending | pending | pending |

Required output tables:

- `thesis_exp/outputs/exp10_qdpr2_module_ablation/tables/exp10_ablation_metrics_summary.csv`
- `thesis_exp/outputs/exp10_qdpr2_module_ablation/tables/exp10_ablation_module_effects.csv`
