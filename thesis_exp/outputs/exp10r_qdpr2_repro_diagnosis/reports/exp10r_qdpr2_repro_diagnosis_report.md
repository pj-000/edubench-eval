# Exp10R QD-PR2 Reproducibility Diagnosis

## Executive summary

Exp09 formal QD-PR2 and Exp10 `full_qdpr2` are configuration-equivalent for the
substantive training setup: same base model, same QD-B1 initialization checkpoint,
same train/dev/test paths, same pair construction parameters, same max pair counts,
same active loss weights, same epochs/batch settings/learning rate, and the same dev
MAE checkpoint-selection rule.

The observed test low-to-high difference is therefore not explained by a config,
pair-pool, or evaluation-script mismatch. The primary reproducibility difference is
checkpoint selection:

| run | selected epoch | selected global step | test MAE | test QWK | test Acc@5 | test low-to-high |
|---|---:|---:|---:|---:|---:|---:|
| Exp09 formal QD-PR2 | 3 | 237 | 0.4192 | 0.6084 | 0.7549 | 12/31 |
| Exp10 full_qdpr2 | 1 | 79 | 0.4146 | 0.6028 | 0.7776 | 14/31 |

Exp09 selected epoch 3, while Exp10 selected epoch 1. Exp10's selected checkpoint has
slightly better test MAE and Acc@5, but it raises low-score severe overestimation from
12/31 to 14/31.

## Configuration diagnosis

The only config/reporting differences marked as different are wrapper-level labels or
selected checkpoint metadata: run:run_id, run:objective, checkpoint selection:selected_best_epoch,
checkpoint selection:selected_best_global_step, source:config_path. The substantive settings
requested for
reproducibility are matched or equivalent:

- Initial checkpoint: same QD-B1 checkpoint and same base model.
- Data paths: same dataset directory and train/dev/test JSONL paths.
- Seed: same pair sampling seed; trainer seed is the same default value.
- Pair construction: same train/dev pair counts, max pairs per record, margins, and
  low-high weighting.
- Loss: Exp09 has implicit `lambda_point=1.0`; Exp10 records it explicitly as `1.0`.
- Optimization: same 3 epochs, batch size, gradient accumulation, learning rate,
  weight decay, warmup ratio, max length, and checkpoint selection metric.

## Evaluation diagnosis

Both runs use the same selected-best evaluation surface: `pred_label_5 = 1 +
count(prob_gt_k > 0.5)`, with threshold probability columns `prob_gt_1..prob_gt_4`
and logit columns `logit_gt_1..logit_gt_4`.

On test, the true low-score count is 31 in Exp09 and
31 in Exp10. The requested low-to-high counts are:

- Exp09 formal QD-PR2: 12/31
- Exp10 full_qdpr2: 14/31

## Checkpoint-level diagnosis

Local per-epoch checkpoint weights were not available, so this diagnosis cannot
recompute test predictions at every epoch. Dev per-epoch metric history is available,
and selected-best dev/test predictions are available.

The decisive dev rows are:

| run | epoch | global step | dev MAE | dev QWK | dev Acc@5 | dev low-to-high | dev monotonic violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp09 formal QD-PR2 | 3 | 237 | 0.4851 | 0.5600 | 0.7806 | 28/57 | 0.3478 |
| Exp10 full_qdpr2 | 1 | 79 | 0.4869 | 0.5520 | 0.7968 | 28/57 | 0.3496 |

Exp09 and Exp10 share identical dev values at epochs 1 and 2, then differ slightly at
epoch 3. Because checkpoint selection minimizes dev MAE, that small epoch-3 drift
changes the selected checkpoint.

## Pair-pool diagnosis

The pair pool on disk contains 10000 train pairs and 3000 dev
pairs. Exp10 does not have a separate local pair archive; the full run points to the
same deterministic QD-PR2 pair-pool setup. Train pair-pool SHA256 signature:
`682c6affbedcef81c3f4133caf8963ae54e83f7d69ee9dd5085e8af3e17943af`.

## Interpretation

The most likely explanation for 12/31 versus 14/31 is checkpoint-selection drift under
an otherwise matched setup. Exp10 selected an earlier checkpoint (epoch 1) because its
epoch-3 dev MAE was slightly worse than epoch 1; Exp09 selected epoch 3 because epoch 3
was best by dev MAE in that run. Since low-to-high is not the selection metric, a
checkpoint with marginally better MAE can still produce more severe overestimation on
the low-score subset.

## Output files

- `thesis_exp/outputs/exp10r_qdpr2_repro_diagnosis/tables/exp10r_config_diff.csv`
- `thesis_exp/outputs/exp10r_qdpr2_repro_diagnosis/tables/exp10r_metric_diff.csv`
- `thesis_exp/outputs/exp10r_qdpr2_repro_diagnosis/tables/exp10r_checkpoint_metrics.csv`
- `thesis_exp/outputs/exp10r_qdpr2_repro_diagnosis/tables/exp10r_pair_pool_diff.csv`
