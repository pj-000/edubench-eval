# Exp17-D1 Hidden Failure Audit Summary

This is a dev-only diagnostic summary. It does not train a model, read test data, or generate checkpoints.

## 1. Summary of Annotated Cases

- Total annotated cases: `27`
- Validation issues: `0`

## 2. Failure Mode Distribution

| failure_mode | n | rate | rubric linked rate | hidden failure rate | conflict rate |
|---|---:|---:|---:|---:|---:|
| format_violation | 1 | 0.0370 | 1.0000 | 0.0000 | 0.0000 |
| surface_fluent_but_hidden_defect | 4 | 0.1481 | 1.0000 | 1.0000 | 0.0000 |
| insufficient_evidence | 7 | 0.2593 | 1.0000 | 1.0000 | 0.0000 |
| possible_label_conflict | 15 | 0.5556 | 0.0000 | 0.0000 | 1.0000 |

## 3. Rubric Linkage Analysis

- Rubric-linked hidden failure rate: `0.4074`

## 4. Hidden Failure vs Visible Defect Analysis

Use `is_hidden_failure_manual` together with the primary failure mode to separate surface-fluent hidden failures from visible format or task violations.

## 5. Possible Label Conflict Analysis

- Possible label conflict rate: `0.5556`

## 6. Question Group Concentration Analysis

- Max question group rate: `0.7407`

## 7. Trainability Analysis

| trainability | n | rate | recommended action |
|---|---:|---:|---|
| weak_train_signal | 4 | 0.1481 | Use with lower weight or as auxiliary evidence signal. |
| format_auxiliary_signal | 1 | 0.0370 | Use for format/task-constraint auxiliary supervision. |
| pairwise_only | 7 | 0.2593 | Use in matched hard-negative or pairwise separation only. |
| review_only | 15 | 0.5556 | Keep for qualitative analysis; do not train from it. |

## 8. Case-Control Comparison Notes

- Case-control CSV provided: `True`
- Case-control rows: `81`
- Rows with manual notes: `0`

## 9. Decision

- Enter Exp17-A recommendation: `False`
- Reason: rubric_linked_hidden_failure_rate < 0.60; possible_label_conflict_rate > 0.35; strong_or_weak_train_signal_rate < 0.50; WARNING: Failure cases are highly concentrated in one question group; Exp17-A must use train-side weak labels and should avoid question_key-specific features.

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