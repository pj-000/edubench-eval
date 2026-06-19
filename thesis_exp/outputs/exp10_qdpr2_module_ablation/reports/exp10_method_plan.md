# Exp10 QD-PR2 Module Ablation Plan

Status: `scaffold_ready`.

Experiment: `exp10_qdpr2_module_ablation`

Dataset: question-disjoint `question_seed42` human-only A4 rows.

Backbone: `Qwen3-Reranker-0.6B`.

Initialization: QD-B1 low-score weighted ordinal checkpoint.

Ordinal form: independent cumulative logits for `P(y > t | x)`, `t=1,2,3,4`.

This experiment evaluates QD-PR2 loss modules without using RLHF, PPO, GRPO, DPO, test-label
tuning, pair mining from dev/test, or checkpoint selection on test labels.

| ablation | lambda_point | lambda_pair | lambda_anchor | lambda_mono | purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| full_qdpr2 | 1.0 | 0.05 | 0.5 | 0.1 | Complete anchored pairwise ordinal calibration. |
| no_pair | 1.0 | 0.0 | 0.5 | 0.1 | Test whether pairwise boundary learning lowers low-to-high risk. |
| no_anchor | 1.0 | 0.05 | 0.0 | 0.1 | Test whether anchor protects pointwise calibration and MAE/QWK. |
| no_mono | 1.0 | 0.05 | 0.5 | 0.0 | Test whether monotonic regularization controls threshold violations. |
| point_only | 1.0 | 0.0 | 0.0 | 0.0 | Check whether gains come from another pointwise fine-tuning round. |
| no_point_diagnostic | 0.0 | 0.05 | 0.5 | 0.1 | Diagnostic only; not a candidate final method. |

Default run command on server. This uses GPUs `6,7`, assigns an ordered queue to each GPU, and runs
all six ablations including the diagnostic run:

```bash
cd ~/edubench-eval-exp2
./thesis_exp/scripts/run_exp10_qdpr2_module_ablation.sh
```

To force a single GPU:

```bash
cd ~/edubench-eval-exp2
EXP10_GPU_LIST=6 ./thesis_exp/scripts/run_exp10_qdpr2_module_ablation.sh
```

To rerun from scratch instead of skipping completed runs:

```bash
cd ~/edubench-eval-exp2
RESET_RUN_DIR=1 SKIP_COMPLETED=0 ./thesis_exp/scripts/run_exp10_qdpr2_module_ablation.sh
```
