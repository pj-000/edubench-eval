# Exp28A Paper-Protocol Input Audit

## Decision

**PASS WITH REANNOTATION REQUIRED**

The small-paper campaign is locked to the original paper-like split:

- train: 2654
- dev: 664
- test: 2218
- isolation unit: question-answer-metric triple

## Legacy Exp27 Mapping

The existing 361-row teacher audit was selected under `question_seed42`, so it is not a
drop-in training set for the paper protocol:

- 190 rows map to paper train and may be retained only as tagged pilot evidence.
- 48 rows map to paper dev and are quarantined.
- 123 rows map to paper test and are quarantined.
- 0 rows are unmatched.

No held-out label was consumed by this audit. Dev/test files were reduced to sample and
question identities solely for overlap checks.

## Locked Rules

1. Teacher annotation and model adjudication may create supervision only for paper train.
2. The 171 legacy held-out annotations cannot be used to choose prompts, fusion rules, or targets.
3. Dev selects the final data version and checkpoint; test remains outside iterative development.
4. Independent model review is reported as model-review silver, not human gold.

## Next Step

Build a fresh paper-train-only annotation inventory and protocol pilot. The 190 reusable legacy
rows remain explicitly tagged so that later ablations can distinguish reused pilot evidence from
new teacher annotations.
