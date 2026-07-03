# Exp19 Second-Round SFT Training Sync Report

Second-round SFT ablation training completed on the server. This sync only
records lightweight training status; adapters, checkpoints, raw logs, and
prediction files are not committed.

## Completed Runs

| run | adapter dir | global step | last logged loss |
|---|---|---:|---:|
| R1b score-only balanced | `saves/edubench/qwen3-4b/r1_score_only_balanced_lora` | 1125 | 0.0202 |
| R2n reason-score natural | `saves/edubench/qwen3-4b/r2_reason_score_natural_lora` | 1248 | 0.0148 |
| R2c clean reason-score balanced | `saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora` | 1125 | 0.0062 |
| R4b shuffled reason balanced | `saves/edubench/qwen3-4b/r4_shuffled_reason_balanced_lora` | 1125 | 0.0107 |

## Notes

- Training used Qwen3-4B LoRA SFT configs from `exp19_sft_dpo_datasets_seed42/configs`.
- This step is training-only; dev/test evaluation has not been run yet.
- The second-round runner cleaned intermediate `checkpoint-*` directories after successful training.
- Existing first-round intermediate checkpoints are still present on the server and can be removed separately if space is needed.
