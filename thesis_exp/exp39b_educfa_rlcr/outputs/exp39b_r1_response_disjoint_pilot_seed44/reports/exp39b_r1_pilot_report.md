# Exp39B-R1 EduCFA-RLCR Response-Disjoint Pilot

## Decision

- Status: **PROTOCOL_PILOT_NO_GO**
- Stopped at: `plan_validation`
- Plan-valid: `28/60` (required `54/60`)
- Qwen editor, DeepSeek critic, revision, and final verifier were not run.

## Source preflight

- Selected rows / qkeys: `60 / 60`
- Source-ID / answer-hash / assessment-key overlap: `0 / 0 / 0`
- Question-key overlap with Exp39A: `60` (expected and allowed)
- Max character/token Jaccard: `0.4281 / 0.5838`

## Planner rejection reasons

- operator_incompatible: `23`
- low_confidence: `10`
- severe_band_clause_mismatch: `4`
- missing_planner_output: `1`

## Interpretation

The response-disjoint source amendment passed. The frozen planner protocol did not reach its pre-registered compatibility gate.
This is a planner-level protocol failure; no downstream generation-quality claim is made.
The result is not unseen-question validation.

## Compliance

- No GPU or training.
- No dev/test access.
- Raw/private API artifacts remain ignored.
