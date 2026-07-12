# Exp37A failure-evidence qualification report

## Status
- Reference complete: `False`
- Reviewer A/B/C rows: `0/0/0`
- Qwen source: `thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42/private/qwen/p0_holistic_zero_shot/all_train.jsonl`
- This analysis never reads dev/test, trains a student, runs student inference, or calls an API.

## Interpretation
Before complete blind reviews and adjudication, semantic metrics are intentionally unavailable and GO is false.
After completion, interpret Qwen evidence only through the preregistered minority-F1, low-tail-recall, evidence-support, and aligned-vs-shuffled OOF gates.

## Decision
- recommend_new_reason_evidence_training: `False`
- recommend_full_train_score_range_annotation: `False`
- stop_reason_evidence_supervision: `False`

## Boundary
Only paper-like train rows and train-only OOF predictions are allowed. No dev/test labels, predictions, or metrics are used.
