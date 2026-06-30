# Exp17-D1 Hidden Failure Audit

Exp17-D1 is a diagnostic experiment for RQ2. It does not train a model, load
transformers, read test data, generate checkpoints, or write raw predictions.

The goal is to audit Exp17-D0 label-2 high-prediction cases and decide whether
the next training experiment, Exp17-A, should learn a hidden-failure evidence
signal.

## Inputs

Default source files are the lightweight Exp17-D0 CSV outputs:

- `thesis_exp/outputs/exp17_low_score_evidence_diagnosis/tables/label2_l2h_cases.csv`
- `thesis_exp/outputs/exp17_low_score_evidence_diagnosis/tables/matched_high_score_controls.csv`

The task spec also supports copying these files under:

- `thesis_exp/exp17_low_score_evidence/outputs/d0_seed42_dev/`

The `outputs/` directory is gitignored.

## Prepare Annotation Template

```bash
python thesis_exp/exp17_low_score_evidence/diagnostics/prepare_hidden_failure_audit.py \
  --d0-cases thesis_exp/outputs/exp17_low_score_evidence_diagnosis/tables/label2_l2h_cases.csv \
  --d0-controls thesis_exp/outputs/exp17_low_score_evidence_diagnosis/tables/matched_high_score_controls.csv \
  --dev-jsonl thesis_exp/data/splits/question_seed42/dev.jsonl \
  --out-dir thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev \
  --split dev \
  --seed 42
```

Generated files:

- `d1_hidden_failure_annotation_template.csv`
- `d1_question_group_summary.csv`
- `d1_matched_case_control_review.csv`
- `exp17_d1_prepare_report.md`

Fill the annotation template manually and save it as:

```text
thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/d1_hidden_failure_annotation_template_filled.csv
```

## Summarize Filled Annotations

```bash
python thesis_exp/exp17_low_score_evidence/diagnostics/summarize_hidden_failure_audit.py \
  --annotated-csv thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/d1_hidden_failure_annotation_template_filled.csv \
  --case-control-csv thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/d1_matched_case_control_review.csv \
  --out-dir thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/summary \
  --split dev
```

Generated files:

- `d1_failure_mode_summary.csv`
- `d1_trainability_summary.csv`
- `d1_question_group_failure_summary.csv`
- `d1_enter_exp17a_decision.json`
- `exp17_d1_hidden_failure_report.md`

## Exp17-A Decision Rule

The default rule recommends entering Exp17-A when:

- `rubric_linked_hidden_failure_rate >= 0.60`
- `possible_label_conflict_rate <= 0.35`
- `strong_or_weak_train_signal_rate >= 0.50`

If `max_question_group_rate >= 0.70`, Exp17-A may still proceed but must avoid
question-key-specific features and should use train-side weak labels rather than
dev annotations directly.

## Leakage Rules

- Dev-only audit.
- Do not read test data.
- Do not use dev annotations directly as train labels.
- Do not commit outputs, checkpoints, raw predictions, jsonl, npy, npz, or logs.
