# Exp17-D1 Hidden Failure Audit Summary

This is a dev-only diagnostic summary. It does not train a model, read test data, or generate checkpoints.

## 1. Summary of Annotated Cases

- Total annotated cases: `27`
- Validation issues: `0`

## 2. Failure Mode Distribution

| failure_mode | n | rate | rubric linked rate | hidden failure rate | conflict rate |
|---|---:|---:|---:|---:|---:|
| task_constraint_violation | 2 | 0.0741 | 1.0000 | 1.0000 | 0.0000 |
| factual_or_rubric_mismatch | 4 | 0.1481 | 1.0000 | 1.0000 | 0.0000 |
| surface_fluent_but_hidden_defect | 4 | 0.1481 | 1.0000 | 1.0000 | 0.0000 |
| missing_key_point | 13 | 0.4815 | 1.0000 | 1.0000 | 0.0000 |
| insufficient_evidence | 3 | 0.1111 | 1.0000 | 1.0000 | 0.0000 |
| possible_label_conflict | 1 | 0.0370 | 0.0000 | 0.0000 | 1.0000 |

## 3. Rubric Linkage Analysis

- Rubric-linked hidden failure rate: `0.9630`

## 4. Hidden Failure vs Visible Defect Analysis

Use `is_hidden_failure_manual` together with the primary failure mode to separate surface-fluent hidden failures from visible format or task violations.

## 5. Possible Label Conflict Analysis

- Possible label conflict rate: `0.0370`

## 6. Question Group Concentration Analysis

- Max question group rate: `0.7407`

## 7. Trainability Analysis

| trainability | n | rate | recommended action |
|---|---:|---:|---|
| strong_train_signal | 4 | 0.1481 | Use as evidence-positive weak supervision after train-side expansion. |
| weak_train_signal | 20 | 0.7407 | Use with lower weight or as auxiliary evidence signal. |
| format_auxiliary_signal | 1 | 0.0370 | Use for format/task-constraint auxiliary supervision. |
| pairwise_only | 1 | 0.0370 | Use in matched hard-negative or pairwise separation only. |
| downweight_or_exclude | 1 | 0.0370 | Downweight or exclude from evidence-positive labels. |

## 8. Case-Control Comparison Notes

- Case-control CSV provided: `True`
- Case-control rows: `81`
- Rows with manual notes: `0`

## 9. Decision

- Enter Exp17-A recommendation: `True`
- Reason: WARNING: Failure cases are highly concentrated in one question group; Exp17-A must use train-side weak labels and should avoid question_key-specific features.

## 10. Recommended Exp17-A Supervision Design

- Prefer a scalar hidden failure score before multi-class defect type prediction.
- Do not train multi-class defect type unless enough train-side weak labels exist.
- Use human-agreement-weighted weak supervision.
- Ignore or downweight possible label conflict cases.
- Use matched hard negatives only after D1 confirms trainable hidden failure patterns.

## 11. Leakage Statement

- Dev only.
- No test read.
- Dev annotations are not used directly as train labels.
- No checkpoint generated.