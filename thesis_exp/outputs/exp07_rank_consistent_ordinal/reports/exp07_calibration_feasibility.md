# Exp7 Calibration Feasibility

Scope: inventory only. No model training, API calls, synthetic generation, calibration fitting, or
raw prediction/array edits were performed.

Server check: `checked`.

## Inventory

| run_id | dev logits | test logits | dev probs | test probs | dev predictions | test predictions | arrays | checkpoint | calibration ready |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QD-B0_human_only_ordinary_ordinal | yes_server | yes_server | yes_server | yes_server | yes_server | yes_server | yes_server | yes_server | yes_server |
| QD-B1_human_only_L1_weighted_ordinal | yes_server | yes_server | yes_server | yes_server | yes_server | yes_server | yes_server | yes_server | yes_server |
| QD-R1_CORAL_human_only | yes_local | yes_local | yes_local | yes_local | yes_local | yes_local | yes_local | yes_server | yes_local |

## Feasibility Notes

- QD-B0_human_only_ordinary_ordinal: `yes_server`; ready on server; sync arrays/predictions locally or run calibration on server.
- QD-B1_human_only_L1_weighted_ordinal: `yes_server`; ready on server; sync arrays/predictions locally or run calibration on server.
- QD-R1_CORAL_human_only: `yes_local`; ready in local workspace.

## Recommendation

- Do not start Exp7-B yet; the next step is risk-aware ordinal calibration feasibility/export planning.
- QD-R1 calibration is possible from the available dev/test logits and probabilities, but the raw QD-R1 scorer overestimates low scores.
- QD-B0/QD-B1 local calibration is not ready unless their logits/probs/arrays/predictions are synced locally.
- Because the server has the required artifacts, sync them or run the calibration workflow on the server.
- If a local-only workflow is required, export logits/probs/predictions eval-only from the available checkpoints.
