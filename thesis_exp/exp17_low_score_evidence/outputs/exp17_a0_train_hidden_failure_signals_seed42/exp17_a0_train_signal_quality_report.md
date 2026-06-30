# Exp17-A0 Train Hidden Failure Signal Quality Report

This is a train-only signal construction diagnostic. It does not train a model, read test data, or use dev D1 annotations as train labels.

## Inputs

- Train split: `thesis_exp/data/splits/question_seed42/train.jsonl`
- Human rationale files under `5-grades/`.
- Missing reason files: `none`
- Loaded optional D1 taxonomy summary rows: `9`.

## Summary

- total train samples: `3326`
- train low-label samples: `111`
- recovered rationale coverage among low-label samples: `84/111` = `0.7568`
- clean evidence-positive count: `0`
- weak evidence-positive count: `76`
- pairwise_low count: `0`
- format_auxiliary count: `4`
- answer_key_dependent count: `4`
- conflict_or_exclude count: `0`
- unclear count: `27`
- score_phrase_conflict count: `2`
- number of question groups covered by clean positives: `27`
- number of metrics covered by clean positives: `8`
- max question_group rate among positives: `0.0921`
- clean high controls: `2656`
- matched pair count: `420`
- Exp17-A1 recommended: `True`
- Recommendation reason: `all A1 entry rules satisfied`

## Candidate Type Counts

| hidden_failure_candidate_type | n |
|---|---:|
| strong_evidence_positive | 0 |
| weak_evidence_positive | 76 |
| pairwise_low | 0 |
| format_auxiliary | 4 |
| answer_key_dependent | 4 |
| conflict_or_exclude | 0 |
| unclear | 27 |

## Failure Mode Counts

| failure_mode_auto | n |
|---|---:|
| unclear | 27 |
| surface_fluent_but_hidden_defect | 25 |
| missing_key_point | 25 |
| insufficient_evidence | 23 |
| task_constraint_violation | 4 |
| answer_key_or_reference_mismatch | 4 |
| format_violation | 3 |

## Exp17-A1 Entry Rule Check

| check | value | required | pass |
|---|---:|---:|---|
| evidence_positive + weak_evidence_positive | 76 | >= 50 | True |
| question_group_count | 27 | >= 5 | True |
| metric_count | 8 | >= 3 | True |
| max_question_group_rate | 0.0921 | <= 0.50 | True |
| score_phrase_conflict_rate_among_positives | 0.0132 | <= 0.20 | True |
| matched_pair_count | 420 | >= 100 | True |

## Redaction Notice

`train_hidden_failure_candidates.csv` and `train_clean_high_controls.csv` intentionally include raw `question`, `answer`, and recovered human rationale text for auditability. If these artifacts will be shared outside the project, create a redacted copy first.
