# SORC-DPO preference-pair protocol v1

## Scientific question

After RAR-SFT, can policy-observed ordinal score errors be converted into
field-controlled preferences and assigned risk-sensitive margins so that
severe low-to-high errors fall without degrading overall scoring, high-score
recognition, or visible rationale alignment?

## Authoritative data boundary

- Use only the frozen train split.
- Failure evidence comes only from R3 epoch 3, seeds 42/43/44, under the
  completed greedy failure-bank protocol.
- The three outputs for one record are correlated policy evidence. They are
  never treated as three independent preference examples.
- Every future preference-training seed uses byte-identical pair manifests.
- No new stochastic rollout, dev access, test access, or preference training
  is allowed at this checkpoint.

## Score pairs

The chosen score is always `label_5`. The rejected score is selected from
actual R3 errors whenever the protocol permits. Both responses use the same
canonical, already score-redacted R3 human rationale; records without an
active R3 rationale use an empty rationale. Only score-value tokens enter the
score preference loss.

Three blocks are built and averaged with equal block weight:

1. `adjacent_score`: actual errors at distance one; no synthetic backfill.
2. `severe_l2h`: exactly the 76 Label-1/2 records. The frozen bank provides
   53 actual severe errors; the remaining 23 use rejected score 4.
3. `h2l_guard`: 76 non-reused high-score records mirrored from the low block
   (24 Label-5 and 52 Label-4). Actual underestimates of any distance are used
   before synthetic rejected score 2. Matching then prefers metric and
   language agreement.

A record contributes at most one pair inside each block. Repeated evidence
from multiple R3 seeds changes provenance and deterministic candidate
selection, never pair weight.

The seed-42 matched-synthetic rejected-score diagnostic uses the same records,
blocks, and label/metric/language strata as the hybrid set. Its record set is
therefore selected with help from the actual failure bank; only its rejected
score is synthetic. It estimates the value of actual versus synthetic
rejected scores conditional on the same policy-selected records, not the
incremental value of the complete record-mining process.

## Rationale pairs

The formal rationale pair uses a frozen R2/R3 active event:

- chosen: R3 aligned human rationale;
- rejected: R2 strict shuffled human rationale;
- score: identical on both sides;
- loss: rationale content tokens only.

One canonical event is selected per record to avoid event, epoch, or seed
pseudoreplication. Rationale bytes must differ and both materialized targets
must be untruncated. P3 remains disallowed until a fixed train-only,
two-evaluator-family, order-swapped aggregate blind qualification favors R3.

## Actual-rationale diagnostic

Actual SFT rationales are not formal P3 ground truth. A seed-42 diagnostic may
be constructed only from complete, correct-score, non-forced, non-leaking
outputs. These become candidates, not preference labels. A pair is eligible
only after two independent evaluator families prefer the aligned R3 human
rationale in both A/B orientations. At least 128 unique records are required;
otherwise the diagnostic is reported but not trained.

## Field-DPO and SORC-DPO

Let `A_i` be the chosen-minus-rejected policy/reference log-probability
contrast using only active field tokens. The matched Field-DPO baseline is:

```text
-log sigmoid(beta * A_i)
```

For score pairs:

```text
C_i     = abs(rejected_score - gold_score) / 4
          + I(gold_score <= 2 and rejected_score >= 4)
offset  = C_i / 2
L_ODPO  = -log sigmoid(beta * A_i - offset)
```

The offset is already in estimated-reward space and is not multiplied by
`beta` again. Rationale pairs use offset zero.

## Formal experiment roles

- P0: frozen R3-SFT baseline.
- P1: Field-DPO on the hybrid score pairs, offset zero.
- P2: SORC-DPO score component on byte-identical pairs.
- P3: full SORC-DPO, adding the qualified R3-vs-R2 rationale block.
- P1-SYN-s42: matched pure-synthetic data-source diagnostic.
- P3-ACT-s42: optional silver actual-rationale diagnostic after qualification.

This protocol supports claims about visible rationale alignment and
policy-observed score-error-conditioned counterfactual preferences. It does
not establish internal reasoning improvement, chain-of-thought faithfulness,
or an unbiased unseen-data failure distribution.

## Candidate construction checkpoint

The train-only candidate build produced:

- 838 hybrid score pairs: 686 adjacent, 76 severe low-to-high, and 76
  high-to-low guard pairs;
- 802 actual-controlled and 36 minimal synthetic-backfill score pairs;
- 838 stratum-matched pure-synthetic seed-42 diagnostic pairs;
- 1,600 canonical R3-aligned versus R2-shuffled rationale pairs;
- 948 eligible actual-model rationale candidates, without assigning them
  preference labels.

The independent pair auditor passed. The formal rationale qualification
package then selected 120 unique pairs with frozen quotas of 24 Label-1/2,
32 Label-3, and 64 Label-4/5 records. It covers all 12 metrics and has 60
English and 60 Chinese records. Each pair has both A/B orientations in both a
score-blind and a score-visible stage, yielding 240 presentations per stage
and 480 presentations per evaluator family. A separate auditor independently
rederived the selection, orientation keys, and task bytes.

This checkpoint freezes neither the candidate pairs nor the qualification
result. Two evaluator families must still complete the qualification, and its
aggregate rule must pass, before P3 can use the rationale block. No
preference training, dev access, or test access is authorized here.

The candidate count was reduced from 975 to 948 after replacing
target-score-only leakage detection with a scan for any explicit 1--5 score
mention. The 838 score pairs and 1,600 formal R3/R2 human-rationale pairs were
unchanged.
