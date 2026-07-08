# Exp27C Teacher Audit V3 Re-Pilot Preparation

This step revises the teacher-audit annotation protocol before scaling. It does not call APIs or
train.

## Counts

- v3 re-pilot packets: 60
- Exp27A overlap rows: 20
- new risk-focused rows: 40
- new low rows: 16
- new label-3 borderline rows: 12
- new high-control rows: 12

## V3 Repair Changes

- Adds education-specific failure enums: `missing_personalization` and `missing_scenario_integration`.
- Strengthens schema-first prompt rules to prevent extra keys, translated keys, and enum drift.
- Clarifies conditional `score_cap` and `no_major_failure` / `overestimation_risk` rules.
- Keeps exact-or-missing evidence discipline: answer spans must be answer substrings; absence failures use null spans plus missing reasons.
- The API runner can retry once with schema-error feedback; local code does not silently repair teacher judgments.

## Guardrails

- Blind packets do not contain original scores.
- Blind packets do not contain recovered human reasons.
- Dev/test are read only for sample_id/question_key leakage guards.
- Test labels are not read.
- No API call or model training is performed in preparation.
