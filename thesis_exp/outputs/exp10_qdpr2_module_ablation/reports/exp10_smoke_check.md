# Exp10 Smoke Check

Status: `PASS`

| ablation | check | status | details |
| --- | --- | --- | --- |
| full_qdpr2 | loss_config.csv exists | PASS | thesis_exp/outputs/exp10_qdpr2_module_ablation/smoke_test/full_qdpr2/tables/loss_config.csv |
| full_qdpr2 | loss_debug_history.csv exists | PASS | thesis_exp/outputs/exp10_qdpr2_module_ablation/smoke_test/full_qdpr2/tables/loss_debug_history.csv |
| full_qdpr2 | metrics_summary.csv exists | PASS | thesis_exp/outputs/exp10_qdpr2_module_ablation/smoke_test/full_qdpr2/tables/metrics_summary.csv |
| full_qdpr2 | run completed | PASS | completed |
| full_qdpr2 | force_pair_training loads | PASS |  |
| full_qdpr2 | use_pair_training resolves | PASS |  |
| full_qdpr2 | dataloader_mode matches | PASS |  |
| full_qdpr2 | lambda_pair matches | PASS |  |
| no_pair | loss_config.csv exists | PASS | thesis_exp/outputs/exp10_qdpr2_module_ablation/smoke_test/no_pair/tables/loss_config.csv |
| no_pair | loss_debug_history.csv exists | PASS | thesis_exp/outputs/exp10_qdpr2_module_ablation/smoke_test/no_pair/tables/loss_debug_history.csv |
| no_pair | metrics_summary.csv exists | PASS | thesis_exp/outputs/exp10_qdpr2_module_ablation/smoke_test/no_pair/tables/metrics_summary.csv |
| no_pair | run completed | PASS | completed |
| no_pair | force_pair_training loads | PASS |  |
| no_pair | use_pair_training resolves | PASS |  |
| no_pair | dataloader_mode matches | PASS |  |
| no_pair | lambda_pair matches | PASS |  |
| no_pair | old no_pair uses pointwise loader | PASS |  |
| no_pair_same_pair_batches | loss_config.csv exists | PASS | thesis_exp/outputs/exp10_qdpr2_module_ablation/smoke_test/no_pair_same_pair_batches/tables/loss_config.csv |
| no_pair_same_pair_batches | loss_debug_history.csv exists | PASS | thesis_exp/outputs/exp10_qdpr2_module_ablation/smoke_test/no_pair_same_pair_batches/tables/loss_debug_history.csv |
| no_pair_same_pair_batches | metrics_summary.csv exists | PASS | thesis_exp/outputs/exp10_qdpr2_module_ablation/smoke_test/no_pair_same_pair_batches/tables/metrics_summary.csv |
| no_pair_same_pair_batches | run completed | PASS | completed |
| no_pair_same_pair_batches | force_pair_training loads | PASS |  |
| no_pair_same_pair_batches | use_pair_training resolves | PASS |  |
| no_pair_same_pair_batches | dataloader_mode matches | PASS |  |
| no_pair_same_pair_batches | lambda_pair matches | PASS |  |
| no_pair_same_pair_batches | strict run uses pair loader | PASS |  |
| no_pair_same_pair_batches | strict lambda_pair is zero | PASS |  |
| no_pair_same_pair_batches | strict weighted_loss_pair is zero | PASS |  |
| no_pair_same_pair_batches | strict total excludes L_pair contribution | PASS | total=0.012214324437081814; recomputed=0.01221432420425117 |
| no_anchor | loss_config.csv exists | PASS | thesis_exp/outputs/exp10_qdpr2_module_ablation/smoke_test/no_anchor/tables/loss_config.csv |
| no_anchor | loss_debug_history.csv exists | PASS | thesis_exp/outputs/exp10_qdpr2_module_ablation/smoke_test/no_anchor/tables/loss_debug_history.csv |
| no_anchor | metrics_summary.csv exists | PASS | thesis_exp/outputs/exp10_qdpr2_module_ablation/smoke_test/no_anchor/tables/metrics_summary.csv |
| no_anchor | run completed | PASS | completed |
| no_anchor | force_pair_training loads | PASS |  |
| no_anchor | use_pair_training resolves | PASS |  |
| no_anchor | dataloader_mode matches | PASS |  |
| no_anchor | lambda_pair matches | PASS |  |
