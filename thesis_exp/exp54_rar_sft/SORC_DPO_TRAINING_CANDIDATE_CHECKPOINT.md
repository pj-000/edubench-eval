# SORC-DPO training implementation candidate

Status: CPU implementation and materialized-manifest candidate only. No
evaluator, model load, GPU training, forward/backward, dev, or test access is
authorized by this checkpoint.

## Frozen input

The reviewed `PAIR_PROTOCOL_PASS` artifacts are promoted by
`pair_protocol_frozen_lock.json`. The frozen private source contains:

- 838 hybrid score pairs: 686 adjacent, 76 severe low-to-high, and 76
  high-to-low guard pairs;
- 838 matched synthetic score pairs for the seed-42 diagnostic;
- 1,600 R3-aligned versus R2-shuffled rationale pairs;
- 948 actual-model rationale candidates that remain diagnostic-only and are
  absent from every training manifest.

## Field-local objective

For the active field only, sequence log-probability is the mean causal
log-probability over that field's tokens. Let

```text
A_i = (log pi(chosen) - log pi(rejected))
      - (log ref(chosen) - log ref(rejected))

L_i = softplus(-(beta * A_i - delta_i))
```

`beta=0.1`. P1 and P1-SYN set every `delta_i=0`. P2 and the score part of P3
use the frozen pair-specific ordinal-risk offset. The offset is subtracted
after beta scaling and is not multiplied by beta again. Rationale pairs use
zero offset.

Score pairs activate only score-value tokens. Rationale pairs activate only
rationale-content tokens. Prompt tokens, the other task field, JSON syntax,
assistant suffix, and padding are inactive.

## Exact block aggregation

The dataset-level score objective is:

```text
(mean(adjacent) + mean(severe_l2h) + mean(h2l_guard)) / 3
```

For a manifest containing `N` pairs, a score row in block `b` receives:

```text
N * score_share / (3 * block_count_b)
```

A rationale row receives:

```text
N * rationale_share / rationale_pair_count
```

The final loss is `sum(weight_i * L_i) / N`. P3 uses
`score_share=rationale_share=0.5`; the score-only arms use `score_share=1`.
The implementation exposes the numerator and the real accumulation-group
denominator separately so the final short group is not normalized as if it
contained 32 pairs.

## Candidate run matrix and budget

| Arm | Data | Objective | Seeds | Pairs | Effective pair batch | Steps |
|---|---|---|---:|---:|---:|---:|
| P1 Field-DPO | hybrid score | equal three-block score, zero offset | 42/43/44 | 838 | 32 | 27 |
| P2 SORC-score | same hybrid score | equal three-block score, risk offset | 42/43/44 | 838 | 32 | 27 |
| P3 Joint SORC | P2 score + R3/R2 rationale | 0.5 score + 0.5 rationale | 42/43/44 | 2,438 | 91 | 27 |
| P1-SYN | same-record matched synthetic score | P1 loss | 42 only | 838 | 32 | 27 |

P1 and P2 have byte-identical pair identities and token materialization; only
the ODPO offset differs. P1-SYN uses the same record/block vector, chosen
sequence, rationale anchor, and chosen field mask as P1; it changes the source
and provenance of the rejected score. P3 retains all 2,438 pairs but uses
91-pair accumulation so all four arms have exactly 27 optimizer updates and
the same scheduler length. P3 remains higher-FLOP because it processes more
chosen/rejected tokens; it is step-matched, not FLOP-matched, against P2.

Shared optimization candidate: one physical preference pass, learning rate
`5e-7`, AdamW, zero weight decay, cosine schedule, 5% warmup, BF16, and
micro-batch one pair. P1/P2/P1-SYN accumulate 32 pairs; P3 accumulates 91.
Policy and reference are the seed-matched frozen R3 epoch-3 checkpoint;
optimizer state starts fresh.

Padding is uniquely frozen as fixed right-padding of both chosen and rejected
to 2,048 tokens. The public forward-token budgets for one physical pass are:

- P1/P2/P1-SYN: `2 * 838 * 2048 = 3,432,448`;
- P3: `2 * 2438 * 2048 = 9,986,048`.

## Validation evidence

- New Field-DPO/SORC-DPO CPU tests: 22 passed.
- Related failure-bank, pair, qualification, loss, and collator regression:
  46 passed.
- Independent materialized-manifest audit:
  `SORC_DPO_TRAINING_CANDIDATE_AUDIT_PASS`.
- Independently verified manifest rows: P1 838, P2 838, P3 2,438, P1-SYN 838.
- Semantic-source, token/mask, weight/offset mismatches: zero.

The next review gate should decide only whether the loss mathematics,
field-mask implementation, exact three-block aggregation, and disclosed
training/token budgets are acceptable. GPU preference training remains
forbidden until that gate passes and a separate train-only smoke is authorized.
