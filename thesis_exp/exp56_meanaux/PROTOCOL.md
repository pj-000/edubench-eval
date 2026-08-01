# Exp56 MeanAux locked dev-only protocol

## Scientific question

Exp56 asks whether the development-set gain of HMSA is specific to fitting the
five-class empirical rater distribution, or whether a separate auxiliary task
using the finer-grained three-rater mean provides a similar benefit.

The current data contain only unanimous ratings or adjacent-label 2:1 ratings.
Within every observed split, the empirical distribution and continuous human
mean are exactly one-to-one.  Exp56 therefore compares target
representations and training objectives; it cannot establish that one target
contains more label information than the other.

## Fixed method

- Backbone, tokenizer, input, hard head, optimizer, scheduler, epochs, batch
  size, seeds, checkpoint rule, and hard-head inference match Exp51.
- The auxiliary head is a deep copy of the initialized five-class hard head,
  exactly as in HMSA.  It has the same parameter count and introduces no new
  initialization draw.
- Its scalar prediction is the expected score under its five logits:
  `sum(softmax(mean_logits)[k] * (k + 1))`.
- The auxiliary target is `human_mean_5`.
- Loss:
  `CE(hard_logits, label_5) + 1.0 * SmoothL1(expected_mean, human_mean_5)`.
- SmoothL1 beta is fixed at `1.0`.
- No lambda, beta, epoch, or checkpoint-rule search is permitted.
- Checkpoint selection uses hard-head development Exact Match only; ties keep
  the earlier epoch.
- All reported scoring metrics and predictions use hard logits only.

## Evidence stages

1. CPU unit tests and a real-model preflight must verify initialization,
   hard-path parity, gradient routing, and test isolation.
2. A small GPU smoke may run only after explicit user authorization.
3. Seed 42 is the only initial scout.  It is compared with the frozen paired
   Hard-only seed 42 under the same gate used for Exp51.
4. Seeds 43 and 44 may run only if seed 42 passes that gate.
5. Exp56 is train/dev-only.  Source code must reject `test`, and no old test
   prediction is permitted.

## Interpretation

- HMSA better than MeanAux: evidence that distributional target geometry/loss
  is more effective in this setting, not that it contains additional
  information.
- Similar results: evidence that finer-grained human supervision, rather than
  a distribution-specific mechanism, explains much of the gain.
- MeanAux better than HMSA: HMSA is not the simplest supported formulation and
  the paper must be repositioned.
