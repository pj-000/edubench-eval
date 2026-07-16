# Exp47A Label-2 Identifiability and Generalization Audit

Primary diagnosis: **QUESTION_GENERALIZATION_LIMIT**

## Human label structure

- Hard label-2 rows: 52.
- Stable label-2 rows: 52 (100.00%); strict [2,2,2]: 33.
- Ambiguous label-2 rows: 0 (0.00%).
- Median human score range: 0.0000.

## Concentration

- Unique question keys: 28.
- Effective question keys: 15.91.
- Maximum single-question share: 15.38%.

## Train versus unseen-question behavior

- 4B stable-label2 outer-train recall: 0.9952.
- 4B stable-label2 heldout recall: 0.0000.
- 4B correctly predicts label 2 on 207/208 outer-train fold-sample predictions, versus 0/52 OOF heldout rows.
- 0.6B stable-label2 heldout recall: 0.0000.
- 0.6B outer-train recall is unavailable because its fold checkpoints were removed; it was not recomputed or substituted.
- 4B class-2 top-2 rate on stable train rows: 1.0000; heldout: 0.0577.
- 4B mean class-2 probability on stable train rows: 0.8462; heldout: 0.0235.
- On heldout label-2 rows, 4B predicts class 4 for 21 and class 5 for 22 cases.

## Existing OOF overall metrics

- 0.6B heldout MAE/QWK/Exact: 0.3873 / 0.4649 / 0.6733.
- 4B heldout MAE/QWK/Exact: 0.4043 / 0.4420 / 0.6590.

## Decision

- New independent low-tail human data required: **True**.
- Selective/set-valued scoring recommended: **False**.
- Student KD: **false**.
- Test access: **false**.
- No new training is authorized by this audit.

## Integrity

No model was trained, no API was called, dev/test were not opened, and no row-level prediction, logit, sample-ID, checkpoint, or log artifact is public.
