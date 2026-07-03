# Exp19 First-Round SFT Training Sync Report

Training finished on the server. This report summarizes training logs only; dev/test evaluation has not been run yet.

| Run | Completed | Steps | Epoch | Elapsed | Last logged loss | Adapter dir |
|---|---:|---:|---:|---:|---:|---|
| R1 score-only natural | True | 1248 | 3.0 | 1:36:05 | 0.0425 | `saves/edubench/qwen3-4b/r1_score_only_lora` |
| R2 reason-score balanced | True | 1125 | 3.0 | 1:20:46 | 0.0116 | `saves/edubench/qwen3-4b/r2_reason_score_balanced_lora` |
| R4 shuffled reason control | True | 1125 | 3.0 | 1:23:14 | 0.0144 | `saves/edubench/qwen3-4b/r4_shuffled_reason_lora` |

## Sync Policy

- Synced lightweight shell logs and `trainer_log.jsonl` files to local.
- Did not sync LoRA adapter/checkpoint directories to local because they total about 5.6 GB.
- Server adapters remain available for the next dev inference/evaluation step.
