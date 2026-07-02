# Exp19-R0A Qwen3-4B Direct Scoring Baseline

## Setup

- Dataset: `thesis_exp/data/processed/edubench_scoring_all.jsonl`.
- Split mapping: `thesis_exp/data/splits/question_seed42`.
- Model: `/home/jpang/models/modelscope/Qwen/Qwen3-4B`.
- Backend used: `vllm`.
- Training: none.
- Prompt fixed before evaluation: yes.
- Human rationale, human score, D1 label, failure type gold, and existing judge scores are not included in the prompt.

## Overall Metrics

| split | n | MAE | QWK | Exact | low-to-high | label2 recall | parse success |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all_5536` | 5536 | 0.6443 | 0.1158 | 0.5414 | 0.8040 | 0.0531 | 1.0000 |
| `train` | 3326 | 0.6362 | 0.1267 | 0.5436 | 0.7838 | 0.0377 | 1.0000 |
| `dev` | 1107 | 0.7651 | 0.0433 | 0.4977 | 0.8246 | 0.1053 | 1.0000 |
| `test` | 1103 | 0.5476 | 0.1776 | 0.5784 | 0.8387 | 0.0000 | 1.0000 |

## Answers

- Did the model overestimate low-score samples? all_5536 low-to-high rate is `0.8040`.
- Does it recover label2 recall? all_5536 label2 recall is `0.0531`.
- Existing judge comparison table: `tables/qwen3_4b_compare_existing_judges.csv`.
- Missing existing judge columns: `none`.
- Exp16A/C0b references should be compared from their committed reports; this script does not read model checkpoints.
- Suitable to proceed to failure-type prompting or LoRA? failure-type prompting: `True`; LoRA: `False`.

## Guardrails

- Full internal predictions are written under `predictions/full_predictions_internal.jsonl` and must not be committed.
- Full prompts and raw answers are not written to output artifacts.
