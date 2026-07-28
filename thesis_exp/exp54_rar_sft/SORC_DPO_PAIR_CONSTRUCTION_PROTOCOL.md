# SORC-DPO train-only preference-pair construction

## Status

This document freezes the negative-sample design direction. It does not
authorize preference training or test access. The negative-source decision is
still under scientific review; the actual failure bank is being prepared as a
neutral diagnostic rather than as an already selected preference manifest.

## Core principle

Every candidate is a complete JSON output with the existing Exp54 schema:

```json
{"score": 2, "rationale": "..."}
```

Formal pairs are deterministic counterfactuals built from train labels and
frozen R3/R2 rationale sources. Free-text model rollouts are not the main
negative source. Invalid JSON, truncated text, and obviously broken outputs
are never used as rejected examples because they would teach formatting rather
than educational scoring.

Each pair changes exactly one supervised field:

- score pairs use byte-identical rationales and supervise only score-field
  log-probabilities;
- rationale pairs use identical scores and supervise only rationale-field
  log-probabilities.

No pair may read dev or test.

## Chosen output

For train row \(i\):

- \(y_i\) is the aggregate `label_5`;
- \(r_i\) is one frozen, deterministic R3 label-consistent rationale target,
  or the existing inactive/empty representation when no rationale is active.

The chosen score output is:

```text
chosen_i = {"score": y_i, "rationale": r_i}
```

The exact R3 reference schedule used to select \(r_i\) must be fixed before
materializing pairs. Multiple human references must not silently duplicate a
sample's score-pair weight.

## Score-negative blocks

Before pair materialization, obtain the probability of score values 1–5 from
each frozen R3 seed checkpoint and average the three probability vectors. This
is a five-way score-position forward pass, not free-text rollout. Stable ties
are resolved by smaller score value.

### Adjacent block

Construct one pair for every one of the 2,654 train rows. The rejected score is
the model's most likely adjacent wrong label:

| Gold score | Candidate rejected scores |
|---:|---|
| 1 | 2 |
| 2 | 1, 3 |
| 3 | 2, 4 |
| 4 | 3, 5 |
| 5 | 4 |

The rejected output is:

```text
rejected_i_adj = {"score": most_likely_adjacent_error, "rationale": r_i}
```

### Severe low-to-high block

For every Label-1/2 train row, add one mandatory hard negative. There are 76
such rows in the frozen train split. The rejected score is whichever of 4 or 5
has the larger frozen three-seed mean probability:

```text
rejected_i_l2h = {"score": argmax_{s in {4,5}} p_mean_i(s),
                  "rationale": r_i}
```

These are not allowed to disappear inside ordinary class-frequency sampling.
The L2H block receives its own normalized loss contribution.

### High-score protection block

For every Label-4/5 train row, add one countervailing hard negative. The
rejected score is whichever of 1 or 2 has the larger frozen three-seed mean
probability:

```text
rejected_i_h2l = {"score": argmax_{s in {1,2}} p_mean_i(s),
                  "rationale": r_i}
```

This block prevents a trivial reduction in low-to-high errors by shifting the
entire model toward low scores.

## Rationale-negative block

For every existing R2/R3 control-eligible train event:

```text
chosen   = {"score": y_i, "rationale": frozen R3 aligned rationale}
rejected = {"score": y_i, "rationale": frozen R2 shuffled rationale}
```

The R2 donor must retain the already frozen constraints: same label, same
metric, same language, different record/content, no donor reuse within its
control permutation. Score bytes are identical.

When one sample has multiple rationale references, compute reference-level
losses first and average within sample before averaging across samples. This
prevents samples with more references from receiving greater weight.

## Block-balanced objectives

The score preference objective averages internally within three blocks and
then gives each block equal weight:

```text
L_score = (L_adjacent + L_L2H + L_H2L_guard) / 3
```

The full score+rationale method uses:

```text
L_joint = L_score + L_rationale
```

P1 uses the same score-pair manifest with equal pair strength. P2 uses the same
manifest and introduces ordinal/L2H pair-specific strength. P3 adds the frozen
rationale block to P2. Pair data, sampling, optimizer steps, and cold-start
checkpoint remain otherwise matched.

## Ordinal risk

The proposed deterministic risk is:

```text
C(s, y) = |s - y| / 4 + I[y <= 2 and s >= 4]
g(s, y) = C(s, y) / 2
```

Thus nearby errors receive a smaller margin, severe L2H receives the largest
margin, and H2L remains explicitly protected by both ordinal distance and its
own block.

The ODPO paper defines the estimated reward as
`r_hat = beta * log(policy/reference)` and its Eq. 7c subtracts the offset from
the difference of those already beta-scaled estimated rewards. Therefore, if
`Delta` in our implementation is the unscaled chosen-minus-rejected
policy/reference log-ratio difference, the direct ODPO form is:

```text
-log sigmoid(beta * Delta - g)
```

It is not:

```text
-log sigmoid(beta * (Delta - g))
```

unless the quantity inside the parentheses is explicitly defined as
`g / beta`. These forms are not equivalent when `beta != 1`.

Primary source: Amini, Vieira, and Cotterell, *Direct Preference Optimization
with an Offset*, Findings of ACL 2024, Eq. 7c and Eq. 12:
https://aclanthology.org/2024.findings-acl.592/

## Pair hard filters

Reject a candidate pair if any condition holds:

- chosen and rejected are identical;
- the intended field does not differ;
- the protected field is not byte-identical;
- either output is invalid JSON;
- either changed field would be truncated;
- the rationale contains explicit score leakage under the frozen leakage rule;
- an R2 donor violates its frozen matching constraints;
- normalized R2 and R3 rationales are identical;
- provenance is missing;
- any source comes from dev or test.

## Concrete examples

Label-2 severe overestimate:

```json
chosen:
{"score": 2, "rationale": "回答提到了基本结论，但遗漏了 rubric 要求的关键推导。"}

rejected:
{"score": 4, "rationale": "回答提到了基本结论，但遗漏了 rubric 要求的关键推导。"}
```

Label-5 high-score protection:

```json
chosen:
{"score": 5, "rationale": "回答覆盖关键概念，推导完整，满足该维度最高等级要求。"}

rejected:
{"score": 2, "rationale": "回答覆盖关键概念，推导完整，满足该维度最高等级要求。"}
```

Rationale semantic pair:

```json
chosen:
{"score": 2, "rationale": "当前答案遗漏了必要推导，因此只能获得较低评分。"}

rejected:
{"score": 2, "rationale": "该回答覆盖了主要概念，解释充分且结构完整。"}
```

The last rejected rationale must come from the frozen R2 donor map, not from an
ad-hoc author rewrite.
