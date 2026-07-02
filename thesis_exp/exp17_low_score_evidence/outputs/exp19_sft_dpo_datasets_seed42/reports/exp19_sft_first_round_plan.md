# Exp19 First-Round SFT Plan

This first SFT round trains only three Qwen3-4B LoRA adapters:

- R1 score-only natural: score-only baseline on the original train distribution.
- R2 reason-score balanced: main structured supervision with risk-aware balanced sampling.
- R4 shuffled reason control: control for whether structured reason supervision is meaningful.

R3 rationale SFT and R5 DPO are intentionally delayed. R3 adds generation complexity, and DPO should start
from a usable SFT adapter.

## Training Datasets

| Run | Dataset | Count | Purpose |
|---|---:|---:|---|
| R1 | `edubench_r1_score_only_train` | 3326 | Score-only SFT baseline |
| R2 | `edubench_r2_reason_score_balanced_train` | 3000 | Main balanced structured supervision |
| R4 | `edubench_r4_shuffled_reason_control_train` | 3000 | Balanced shuffled reason/failure control |

The original train distribution is high-score dominated. Balanced datasets are training sampling
strategies only; final evaluation must use the original question-disjoint dev/test distributions.

## Shared Hyperparameters

| Hyperparameter | Value |
|---|---:|
| Model | `/home/jpang/models/modelscope/Qwen/Qwen3-4B` |
| Stage | `sft` |
| Finetuning type | `lora` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA target | `all` |
| Cutoff length | 4096 |
| Per-device train batch size | 2 |
| Gradient accumulation steps | 4 |
| Learning rate | `1.0e-4` |
| Epochs | 3 |
| Scheduler | `cosine` |
| Warmup ratio | 0.05 |
| Precision | `bf16` |
| Gradient checkpointing | `true` |

With one GPU per run, the effective batch size per run is `2 * 4 = 8`.

## Run Command

```bash
cd ~/edubench-eval-exp2
./thesis_exp/scripts/run_exp19_sft_first_round.sh
```
