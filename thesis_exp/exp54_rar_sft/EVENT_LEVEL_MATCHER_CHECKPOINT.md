# Event-level R2 matcher candidate checkpoint

## Authorization boundary

The shared base schedule received `SCHEDULE_PASS` at commit
`c9aaf185207b99eb4212914d1b50145a7e802cd2`. This checkpoint implements the
authorized event-level matcher, independent Oracle, and candidate R2/R3 event
mask. It does not build or freeze final training manifests and does not
authorize smoke or formal training.

## Matching contract

Matching is performed independently within the structured tuple:

```text
seed × epoch_index × label_5 × metric_id × language
```

Every active recipient event:

- receives one donor event from a different record and normalized QA;
- is used exactly once as a donor event;
- belongs to the same active recipient/donor set;
- has indegree and outdegree one;
- belongs to a verified nontrivial cycle.

The lexicographic objective remains maximum active-event coverage, then
minimum frozen-tokenizer length difference, then stable event-ID tie-breaking.

## Independent Oracle

- Status: `R2_EVENT_MATCHER_ORACLE_PASS`
- Fixed adversarial cases: 7
- Deterministic random cases: 64
- Maximum exhaustive size: 8
- Input-order shuffles per case: 20
- Objective failures: 0
- Input-order failures: 0

The Oracle enumerates active subsets and permutations independently of the
Hungarian production solver.

## Frozen-tokenizer candidate result

All three formal seeds use 7,962 base row events:

| Quantity | Per seed |
|---|---:|
| Rationale-eligible events | 4,836 |
| Active rationale events | 4,803 |
| Inactive eligible events | 33 |
| Score-only events | 3,126 |
| Active eligible coverage | 99.3176% |
| Event strata | 285 |
| Strata with deactivation | 33 |

Coverage is identical across seeds:

| Label | Eligible events | Active events | Inactive | Coverage |
|---:|---:|---:|---:|---:|
| 1 | 63 | 54 | 9 | 85.71% |
| 2 | 120 | 102 | 18 | 85.00% |
| 3 | 465 | 462 | 3 | 99.35% |
| 4 | 1,767 | 1,764 | 3 | 99.83% |
| 5 | 2,421 | 2,421 | 0 | 100.00% |

The mean absolute token-length difference is 2.461, 2.432, and 2.401 for
seeds 42, 43, and 44. The corresponding maxima are 75, 74, and 67.

## Frequency and mask checks

For every seed, every individual epoch, and every cumulative checkpoint
prefix:

- rationale bytes Counter equality: exact;
- rationale token-ID Counter equality: exact;
- frequency L1 difference: 0;
- supervised rationale-token totals: exact.

Each epoch has 1,601 active rationale events. Each full three-epoch run has
4,803.

The R2 and R3 event-mask files are byte-identical within each seed. Masks
contain no score-deactivation field. All 7,962 base events remain available
for score supervision; the 3,159 rationale-inactive events comprise 3,126
original score-only events and 33 strict-control-ineligible events.

## Current gate

Status:

```text
R2_EVENT_DONOR_MAP_CANDIDATE_READY
R2_R3_EVENT_MASK_CANDIDATE_READY
```

Allowed next action after review: build candidate materialized S0/R1/R2/R3
manifests and audit final serialized bytes, token IDs, loss masks, and training
budgets.

Still prohibited:

- marking any manifest frozen or `READY`;
- smoke training;
- formal training;
- dev/test access for protocol or matching decisions.
