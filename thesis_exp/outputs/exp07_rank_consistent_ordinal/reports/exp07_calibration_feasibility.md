# Exp7 Calibration Feasibility

Scope: inventory only. No model training, API calls, synthetic generation, calibration fitting, or
raw prediction/array edits were performed.

Server check: `not_checked`.

## Inventory

| run_id | dev logits | test logits | dev probs | test probs | dev predictions | test predictions | arrays | checkpoint | calibration ready |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QD-B0_human_only_ordinary_ordinal | no | no | no | no | no | no | no | no | no |
| QD-B1_human_only_L1_weighted_ordinal | no | no | no | no | no | no | no | no | no |
| QD-R1_CORAL_human_only | yes_local | yes_local | yes_local | yes_local | yes_local | yes_local | yes_local | no | yes_local |

## Feasibility Notes

- QD-B0_human_only_ordinary_ordinal: `no`; missing dev_logits, test_logits, dev_probs, test_probs, dev_labels, test_labels, dev_record_ids, test_record_ids, dev_predictions, test_predictions, arrays; no checkpoint found for eval-only export.
- QD-B1_human_only_L1_weighted_ordinal: `no`; missing dev_logits, test_logits, dev_probs, test_probs, dev_labels, test_labels, dev_record_ids, test_record_ids, dev_predictions, test_predictions, arrays; no checkpoint found for eval-only export.
- QD-R1_CORAL_human_only: `yes_local`; ready in local workspace.

## Recommendation

- Do not start Exp7-B yet; the next step is risk-aware ordinal calibration feasibility/export planning.
- QD-R1 calibration is possible from the available local dev/test logits and probabilities, but the raw QD-R1 scorer overestimates low scores.
- QD-B0/QD-B1 local calibration inputs are missing in this workspace; run with server inventory enabled or sync/export the baseline logits before calibration.
