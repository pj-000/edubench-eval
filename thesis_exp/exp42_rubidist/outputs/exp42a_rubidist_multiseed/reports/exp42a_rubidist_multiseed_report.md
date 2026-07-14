# Exp42A RubiDist multiseed report

- Final status: **FACTORIAL_STOP**
- Formal matrix: 4 variants x 5 question-key folds x 3 seeds = 60 runs.
- All models used fixed final epoch 10; no checkpoint selection was performed.
- Main rubric gain gate: `false` (MAE delta=0.000879; QWK delta=0.006979).
- Soft-target non-inferiority gate: `true`.
- Overall/tail guard gate: `true`.
- Three-seed stability gate: `false`.
- Secondary raw-rubric-only signal: `false`.
- Recommend final dev campaign: `false`.
- Stop positive small-paper route: `true`.

## Boundaries

- No API or teacher labeling was used.
- Training used standard hard/soft cross-entropy only, with no custom loss or sample weighting.
- No paper-like dev/test data or historical evaluation predictions were accessed.
- Exp41 post-fit audit is descriptive and does not affect this decision.
