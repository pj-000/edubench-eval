# Exp27D Teacher Audit V4 Re-Pilot Preparation

This step prepares the v4 consensus-calibration protocol before scaling. It does not call APIs or
train.

## Counts

- v4 re-pilot packets: 80
- Exp27C paired overlap rows: 60
- new stress rows: 20
- new low hard-disagreement stress rows: 8
- new high-control disagreement stress rows: 4
- new label-3 borderline stress rows: 4
- new education-dimension stress rows: 4
- fallback stress rows: 0

## V4 Consensus-Calibration Changes

- Blind schema adds `surface_plausibility`.
- Collector derives `failure_bucket` and `derived_overestimation_risk` instead of trusting raw risk alone.
- Audit schema no longer asks the teacher to copy the blind object; it references blind id/hash only.
- Audit schema separates soft/hard strictness and leniency disagreements.
- Exact-or-missing evidence discipline is retained.

## Guardrails

- Blind packets do not contain original scores.
- Blind packets do not contain recovered human reasons.
- All packets are train-only; dev/test are excluded from packet construction.
- Dev/test are read only for sample_id/question_key leakage guards.
- Test labels are not read.
- No API call or model training is performed in preparation.
