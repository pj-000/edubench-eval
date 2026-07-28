# Exp54 rationale blind-audit preregistration

## Status and purpose

Status: `RATIONALE_BLIND_AUDIT_PROTOCOL_PREREGISTERED`.

This document fixes the rationale-evaluation question and analysis rules after
the SFT dev checkpoint selection was frozen, but before any blind judgments are
collected. It does not authorize blind-audit execution, test access, checkpoint
reselection, decoder changes, or preference training.

The primary question is whether R3's answer-aligned, label-consistent human
rationale supervision produces visibly better scoring rationales than R2's
strict shuffled-rationale control. R3 versus R1 is secondary and asks whether
label-consistent reference filtering adds value relative to all-rater
supervision.

The result is called **model-based rationale agreement/preference**. It is not
human correctness, expert correctness, rationale accuracy, or evidence that the
visible rationale faithfully reveals hidden model reasoning.

The machine-readable source of truth is
`configs/rationale_blind_audit_preregistration.json`.

## Frozen checkpoint population

- Arms: R1, R2, and R3.
- Seeds: 42, 43, and 44.
- Checkpoint: logical epoch 3 for every arm and seed.
- Split: dev only.
- Checkpoint-selection lock SHA-256:
  `364a2c9ee023d1b0a2d803530d7898a347a150dd89dbaaa388aa44677cd71df4`.
- The frozen 256-token output budget and current forced-close behavior remain
  unchanged.

The blind audit cannot change the selected epoch, arm definition, output
budget, parser, or decoder. It cannot be used to choose a new SFT checkpoint or
SFT hyperparameter.

## Fixed sample

The private sample manifest contains 120 unique dev rows and is reused for all
three seeds and all compared arms:

1. Include every available Label-1 and Label-2 dev row (expected total: 20).
2. Select the remaining 100 rows deterministically from Labels 3-5 while
   balancing metric, language, and the Label-3 versus Label-4/5 bands.
3. Require coverage of all 12 metrics and both languages.

The selector may use only record identity/order and pre-existing dev metadata:
`label_5`, `metric_id`, and `language`. It must not inspect arm identity,
generated score, generated rationale, forced-close status, or judge
preferences. The selector seed is
`exp54-rationale-blind-audit-sample-v1|20260728`.

Row-level sample identities remain private. Public artifacts contain only
counts, coverage summaries, hashes, and aggregate results.

## Blind paired presentation

Each comparison uses the same record and seed on both sides:

- Primary: R3 versus R2.
- Secondary: R3 versus R1.

With 120 rows and three seeds, each comparison has 360 paired instances before
order replication. Every pair is shown twice, once in each A/B orientation.
The initial orientation is deterministically hash-randomized. If the two
orientations yield inconsistent preferences after mapping back to arm
identity, the final verdict for that pair is `tie`.

The evaluator receives the question, answer, metric definition, rubric, and
each candidate's generated score and visible rationale. It does not receive
the arm, seed, gold label, individual human scores, human reasons,
forced-completion flag, checkpoint identity, or rationale provenance.

The fixed judgment dimensions are:

- metric alignment;
- rubric relevance;
- answer grounding;
- score-rationale consistency;
- specificity;
- unsupported claims.

Each dimension and the overall judgment must be one of `A`, `B`, or `tie`.

## Evaluators

Two independent evaluator model families are required. Neither may have been
used to construct the Exp54 training rationales or donor mappings. Before
execution, the following must be frozen:

- exact model family and model identity;
- immutable revision or endpoint version;
- prompt bytes and prompt hash;
- output schema and parser;
- decoding parameters and runtime;
- private sample-manifest hash;
- deterministic presentation-order hash.

The present preregistration intentionally leaves evaluator identities unset.
Consequently, blind-audit execution remains forbidden until a separately
audited execution package freezes those details.

## Forced-completion handling

The primary analysis includes the actual frozen output for every sampled pair,
including forced-completed outputs. Removing them from the primary result
would create arm-dependent post-hoc selection because the forced-close rates
differ by arm.

Forced-completion status is hidden from evaluators. It is used only for the
preplanned secondary strata:

- neither candidate forced;
- R3 only forced;
- comparator only forced;
- both candidates forced.

A complete-case result may be reported only as a sensitivity analysis. It
cannot replace the primary result, justify regenerating dev, or select a
checkpoint.

## Analysis

Report R3 wins, ties, and losses with ties retained, paired uncertainty
intervals, and cross-evaluator agreement. Report each evaluator family
separately; pooling is secondary.

The primary inferential result is the R3-versus-R2 overall preference.
R3-versus-R1, individual dimensions, forced-completion strata, and pooled
evaluator results are secondary or diagnostic.

Use a paired hierarchical bootstrap with 10,000 replicates and seed 20260728.
Resample the 120 record clusters and carry all three seed-specific pairs for a
selected record into each replicate.

## Next authorization gate

Before any evaluator call, build and freeze the deterministic private sample
manifest, choose and freeze two eligible evaluator families, materialize the
paired prompts, and run an independent candidate audit. Until that gate passes:

- blind-audit execution is not allowed;
- test access is not allowed;
- preference training is not allowed.
