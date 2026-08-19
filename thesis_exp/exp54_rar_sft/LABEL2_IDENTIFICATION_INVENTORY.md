# Label-2 identification audit: artifact inventory

## Outcome

The no-training inventory is complete. Existing evidence is sufficient for
rater-disagreement, support, pair-graph, and parsed-transition analyses. It is
not sufficient for decoder, prior, calibration, or probability-mass-flow
analysis because the frozen dev prediction rows do not contain five-way score
option probabilities.

No GPU was used, no training was started, and no test artifact was read.

## Available evidence

| Evidence | Availability | Key result |
|---|---|---|
| Locked train/dev with three raw rater scores | Local | Complete for every row |
| Label-2 primary dev population | Local | 14 records across 10 question groups |
| R3 epoch-3 dev predictions, seeds 42/43/44 | Server | 664 rows per seed, hash locked |
| P1 LR=5e-6 dev predictions, seeds 42/43/44 | Server | 664 rows per seed, hash locked |
| R3 and P1 frozen adapters | Server | All six present, hash locked |
| Train actual failure bank | Server | 7,962 private rows |
| P1 score-pair graph | Server | 838 private rows |
| Five-way score option probabilities | Neither | Must be extracted from frozen checkpoints |
| R3/P1 backbone OOF checkpoints | Neither | Not required; dev is question-disjoint and the diagnostic calibrator will be cross-fitted by question |

The locked dev data contain 14 Label-2 records. Five satisfy the protocol's
automatic rater-ambiguity screen. This is an early descriptive count only; it
does not yet assign causes to model failures.

## Important correction to the working hypothesis

The P1 pair graph is not globally missing the 2-versus-3 boundary. It contains
22 direct pairs with chosen score 2 and rejected score 3. Therefore the audit
must not claim that P1 never saw this comparison.

The remaining questions are narrower and testable:

- Do those 22 pairs cover the affected records or their metric/language strata?
- Are they too sparse relative to other transitions?
- Did P1 increase score-2 probability but leave score 3 as the argmax?
- Is the apparent failure better explained by prior, calibration, support, or
  measurement ambiguity?

## Server provenance caution

The server artifact worktree was observed at commit
`4e95b0a4f1b41a0e7c43bbaf004e8561d136dc68`, while the frozen audit protocol
was committed at `d255ffb`. Existing historical artifact hashes are recorded
and may be read. New extraction code must not run from the observed server
checkout until the exact reviewed code is synchronized and validated.

## Blocking artifact and execution boundary

The missing artifact is a row-level table containing the conditional
log-probability of each canonical score value 1–5 under the identical frozen
prompt and score-field prefix for:

- R3 epoch 3, seeds 42/43/44;
- P1 Field-DPO LR=5e-6, seeds 42/43/44;
- all 664 locked dev rows.

Producing it requires frozen-model forward passes on a GPU but no training,
backward pass, optimizer step, or test access. GPU execution remains disabled
until the user confirms that a suitable device is idle.
