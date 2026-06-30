# Exp17-D1 Reannotation with Human Rationales

This is a dev-only diagnostic reannotation. It uses recovered original human rating rationales where available and does not train or read test data.

## Summary

- Total D1 cases: `27`
- Cases with exact recovered metric-level rationale: `21/27`
- Cases inferred from same question group/answer pattern: `5/27`
- Possible label conflict cases after rationale recovery: `1/27`
- Strong/weak train-signal cases: `24/27`

## Interpretation

- Recovering the original human reasons changes D1 materially: most label-2 high-prediction cases are no longer unexplained label conflicts.
- The dominant evidence types are missing key duties, shallow correction reasoning, weak scenario adaptation, weak clarity/inspiration, and factual mismatch in the Annales item.
- The evidence is still concentrated in one marketing-manager question group, so dev annotations should not be used directly as training labels.
- Exp17-A, if run, should construct train-side weak labels or matched hard-negative pairs from the same evidence pattern rather than memorizing dev question keys.
