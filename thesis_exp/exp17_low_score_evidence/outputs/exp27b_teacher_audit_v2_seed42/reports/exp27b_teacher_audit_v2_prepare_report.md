# Exp27B Teacher Audit V2 Re-Pilot Preparation

This step revises the teacher-audit annotation protocol before scaling. It does not call APIs or
train.

## Counts

- v2 re-pilot packets: 60
- Exp27A overlap rows: 20
- new risk-focused rows: 40
- new low rows: 16
- new label-3 borderline rows: 12
- new high-control rows: 12

## V2 Changes

- Replaces one overloaded `risk_flag` with `score_region`, `failure_visibility`, and `overestimation_risk`.
- Adds `evidence_type` and `missing_evidence_reason` for missing-content failures.
- Moves answer-key and label-conflict issues out of blind `major_failures`.
- Adds audit-side `label_noise_type`, `recommended_training_use`, and `sample_weight_suggestion`.

## Guardrails

- Blind packets do not contain original scores.
- Blind packets do not contain recovered human reasons.
- Dev/test are read only for sample_id/question_key leakage guards.
- Test labels are not read.
- No API call or model training is performed in preparation.
