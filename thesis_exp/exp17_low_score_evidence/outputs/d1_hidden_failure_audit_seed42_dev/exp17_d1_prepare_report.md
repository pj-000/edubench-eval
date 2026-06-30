# Exp17-D1 Hidden Failure Audit Preparation Report

This script prepares a manual audit template only. It does not train a model, load checkpoints, import transformers, read test data, or generate raw predictions.

## Inputs

- D0 cases: `thesis_exp/outputs/exp17_low_score_evidence_diagnosis/tables/label2_l2h_cases.csv`
- D0 controls: `thesis_exp/outputs/exp17_low_score_evidence_diagnosis/tables/matched_high_score_controls.csv`
- dev jsonl: `thesis_exp/data/splits/question_seed42/dev.jsonl`
- train jsonl: `thesis_exp/data/splits/question_seed42/train.jsonl`
- split: `dev`
- seed: `42`

## Join Strategy

- `sample_id; missing=0`

## Case Counts

- Cases: `27`
- Question group rows: `10`
- Largest question group: `14ba3cb00f998348fe1c491eab066379d3bf192b` with `20` cases (0.7407)

## Human Agreement Pattern Distribution

- `2.0/2.0/2.0`: `23`
- `2.0/2.0/3.0`: `4`

## Automatic Format Flags

- JSON requirement flags: `{'yes': 27}`
- Possible format violation flags: `{'no': 26, 'yes': 1}`

## Train Support

- computed from thesis_exp/data/splits/question_seed42/train.jsonl

## Outputs

- `thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/d1_hidden_failure_annotation_template.csv`
- `thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/d1_question_group_summary.csv`
- `thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/d1_matched_case_control_review.csv`
- Annotation guide: `thesis_exp/exp17_low_score_evidence/docs/exp17_d1_annotation_guidelines.md`

## Leakage Statement

- Dev-only diagnostic preparation.
- Test data is not read.
- No checkpoint or raw prediction file is generated.
- Dev annotations are for diagnosis and should not be used directly as train labels.