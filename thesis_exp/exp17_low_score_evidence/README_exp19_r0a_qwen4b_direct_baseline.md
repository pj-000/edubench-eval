# Exp19-R0A Qwen3-4B Direct Scoring Baseline

Exp19-R0A is a frozen instruction-model judge baseline for the full EduBench
human-scored audit corpus. It uses Qwen3-4B with one fixed direct scoring prompt
and asks the model to output only:

```json
{"score": <integer from 1 to 5>}
```

## Purpose

This experiment answers a baseline question before any reason-aware training:

Can an instruction model directly score EduBench answers better than the ranker
and existing LLM judge baselines, especially on low-score samples and label 2?

## Guardrails

- No model training.
- No LoRA.
- No failure-type training.
- No rationale generation objective.
- No risk gate.
- No prompt tuning.
- The prompt must not include human score, human rationale, D1 labels, failure
  type labels, or existing judge scores.
- Full internal predictions are written to
  `predictions/full_predictions_internal.jsonl` and must not be committed.

## Data

- Full dataset:
  `thesis_exp/data/processed/edubench_scoring_all.jsonl`
- Split mapping:
  `thesis_exp/data/splits/question_seed42/{train,dev,test}.jsonl`
- Reported splits:
  `all_5536`, `train`, `dev`, `test`

## Server Model

The current server has the model at:

```bash
/home/jpang/models/modelscope/Qwen/Qwen3-4B
```

The run script defaults to A6000 GPU 1:

```bash
cd ~/edubench-eval-exp2
./thesis_exp/scripts/run_exp19_r0a_qwen4b_direct_baseline.sh
```

For Qwen3 models, the script disables thinking by default so the model can
return the required short JSON score. Set `ENABLE_THINKING=1` only for a
separate diagnostic run, not for the fixed R0A direct baseline.

## Dry Run

Dry run does not load the model. It tests parsing, split mapping, tables, report,
and decision output on five examples:

```bash
python thesis_exp/exp17_low_score_evidence/run_exp19_r0a_qwen4b_direct_baseline.py \
  --dry_run \
  --max_examples 5 \
  --out_dir /tmp/exp19_r0a_dryrun \
  --overwrite
```

## Outputs

Committed lightweight outputs after a full run:

- `tables/qwen3_4b_direct_overall_metrics.csv`
- `tables/qwen3_4b_low_score_bias.csv`
- `tables/qwen3_4b_per_label_accuracy.csv`
- `tables/qwen3_4b_confusion_matrix_counts.csv`
- `tables/qwen3_4b_confusion_matrix_row_normalized.csv`
- `tables/qwen3_4b_by_metric.csv`
- `tables/qwen3_4b_by_language.csv`
- `tables/qwen3_4b_compare_existing_judges.csv`
- `tables/qwen3_4b_d1_hidden_eval.csv`
- `reports/exp19_r0a_qwen4b_direct_baseline_report.md`
- `decision/exp19_r0a_decision.json`

Do not commit:

- `predictions/full_predictions_internal.jsonl`
- full prompts
- raw answers outside committed aggregate reports
- checkpoints or logs

## Decision Rule

`direct_baseline_success = true` only if:

- parse success on all_5536 is at least 0.95;
- low-to-high is clearly lower than existing judge or Exp16A/C0b references;
- label2 recall is greater than 0.

Otherwise Exp19-R0A remains a reference baseline and should be followed by a
failure-first prompt or reason-aware training branch.
