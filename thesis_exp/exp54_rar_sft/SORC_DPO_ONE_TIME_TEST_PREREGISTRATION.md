# SORC-DPO one-time test preregistration

## Purpose

The one-time test is not a contest to select the best dev arm. It is the
confirmatory component ablation for the frozen preference study:

| Contrast | Question |
|---|---|
| P1 − P0 | Does actual-error-driven field-local DPO add value over R3-SFT? |
| P2 − P1 | Does the ordinal-risk offset add independent value? |
| P3 − P2 | Does adding the rationale block together with its additional token/FLOP budget add bundled score-side value? |

All four arms and seeds 42/43/44 must run. No intermediate test result may be
viewed before all 12 runs are complete.

## Frozen model inventory

- P0: the three frozen R3-SFT logical-epoch-3 checkpoints.
- P1/P2/P3: the nine completed LR=`5e-6`, 27-step preference checkpoints.
- Every adapter is bound by its exact SHA-256 in
  `configs/sorc_dpo_one_time_test_preregistration_v1.json`.

No arm, seed, checkpoint, learning rate, beta, offset, block weight, or
rationale weight may change after test access.

## Test-data isolation

The campaign must be independent of the state of any mutable working-tree copy
of the test split, whether that copy is currently clean or modified.

The runner must materialize the exact Git-index blob:

```text
7749c2c0f166186cc840409d64424b5a78e7222a
```

inside an isolated execution directory and verify the blob identity before
the first inference call. Preregistration records only Git metadata; test
content has not been opened.

## Frozen inference

The campaign reuses the completed dev protocol:

- Qwen3-4B-Instruct-2507;
- vLLM 0.10.0 with XGrammar;
- greedy decoding, temperature `0`;
- `max_new_tokens=256`;
- compact single-object JSON;
- the existing budget-aware forced-close policy;
- no rerun because an output is forced-completed, repetitive, short, long, or
  scientifically disappointing.

## Endpoints

Co-primary:

- MAE against `label_5`;
- low-to-high rate: gold Label 1/2 predicted as 4/5.

The corresponding low-to-high count is descriptive only; it is not one of
the six Holm-adjusted inferential tests.

Guardrails:

- Exact Match;
- Kendall tau-b;
- Signed Bias;
- Label-5 Recall;
- high-to-low rate;
- strict parse and forced-completion rates.

Secondary:

- Label-1 and Label-2 correct counts/recalls;
- quadratic weighted kappa.

The corresponding high-to-low count is descriptive only.

NLL, Brier, and ECE are not included because the frozen generation protocol
does not produce a normalized five-class probability vector. Adding a new
probability-extraction path now would create a new, unfrozen measurement
protocol.

## Statistical interpretation

For every arm, compute each metric separately within seeds 42, 43, and 44,
then take the unweighted mean of the three seed metrics. Do not pool the three
seed outputs into a `3N` dataset for Kendall, QWK, recall, or any other metric.

For each contrast, report the paired point estimate and a 10,000-replicate
record-cluster bootstrap interval. Draw one record-index vector with
replacement and apply the same vector to all four arms and all three seeds.
Within each replicate, recompute the metric per seed, average the three seed
metrics, and then form the contrast.

Positive values always mean benefit:

- MAE, L2H, H2L, and absolute Signed Bias: left/baseline minus right/treatment;
- Exact, Kendall, Label-5 Recall, and QWK: right/treatment minus left/baseline.

Each component contrast explicitly names a baseline arm and a treatment arm.
The textual `P1 vs P0`, `P2 vs P1`, and `P3 vs P2` labels do not imply one
universal subtraction order; all inference uses the endpoint-specific benefit
delta above.

For L2H, H2L, or label recall, omit a replicate for that endpoint if its
required label subset is absent. Omit a Kendall/QWK replicate if any seed's
metric is non-finite. Report valid and omitted counts. Fewer than 9,500 valid
replicates makes that endpoint unresolved and unable to support a claim.

The six primary tests—three contrasts crossed with MAE and L2H—form one
family. Two-sided cluster-bootstrap p-values use an add-one correction and are
adjusted with Holm–Bonferroni at familywise `α=0.05`. Unadjusted percentile
95% intervals remain descriptive.

Predeclared smallest effects of interest are:

| Endpoint | SESOI |
|---|---:|
| MAE | 0.01 |
| L2H rate | 0.05 |
| Exact | 0.01 |
| Kendall tau-b | 0.01 |
| Absolute Signed Bias | 0.02 |
| Label-5 Recall | 0.02 |
| H2L rate | 0.02 |
| QWK | 0.01 |

- **Strong support:** at least one co-primary has positive benefit and
  Holm-adjusted `p<0.05`; the other co-primary is not materially harmful; and
  no guardrail is materially harmful.
- **Directional support:** not strong; at least one co-primary is positive;
  the other is greater than its negative SESOI; and no guardrail is materially
  harmful.
- **Approximately zero:** neither primary is Holm-significant and both
  absolute point estimates are below their SESOI.
- **Material harm:** benefit is at or below the negative SESOI and its
  unadjusted 95% interval is entirely below zero.
- **Unsupported:** mixed evidence satisfying none of the preceding rules.

Strict parse below `1.0` for any arm/seed is an operational failure. A forced-
completion increase is a diagnostic harm when it is at least five percentage
points and its unadjusted 95% interval is entirely above zero. Either condition
disqualifies both strong and directional component support.

P3 is not FLOP-matched to P2: both have 27 optimizer steps, but P3 uses 2,438
pairs with 91-pair accumulation, whereas P2 uses 838 pairs with 32-pair
accumulation. Therefore H3 cannot isolate rationale semantics; it estimates
the bundled addition of the rationale block and its extra compute.

Test score changes cannot overturn the completed P2-versus-P3 rationale audit.
Without a positive independent rationale audit, P3 may not be claimed to
improve internal reasoning or visible rationale quality.

## Current gate

This document freezes the candidate scientific protocol only.

```text
test_execution_allowed = false
test_accessed = false
```

Before execution, an independent audit must verify the checkpoint hashes,
clean test-blob materialization, inference runner, output isolation, metrics,
and no-intermediate-inspection campaign wrapper.
