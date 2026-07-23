# R2 strict-control implementation checkpoint

Date: 2026-07-23

## Decision implemented

The reviewed decision is `PASS_STRICT`:

- exact `label_5 × metric_id × language` strata;
- different sample and different normalized question-answer content;
- no donor reuse;
- strict one-to-one permutation over the active subset;
- maximum active coverage first;
- minimum frozen-tokenizer length difference second;
- identical R2/R3 rationale-active masks;
- inactive rationale references retain score supervision.

## Solver proof

The optional-diagonal Hungarian implementation was retained after an independent brute-force
Oracle audit:

- 6 adversarial cases, including an eight-reference case;
- 64 deterministic random cases;
- maximum size 8 for exhaustive Oracle comparison;
- 20 input-order shuffles per case;
- maximum coverage matched the Oracle in every case;
- conditional minimum length cost matched the Oracle in every case;
- canonical donor mapping was invariant to input order.

Status: `R2_MATCHER_ORACLE_PASS`.

## Frozen tokenizer

- Model ID: `Qwen/Qwen3-4B-Instruct-2507`
- Upstream revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- Server snapshot:
  `/home/share/models/modelscope/Qwen/Qwen3-4B-Instruct-2507`
- `tokenizer.json` SHA-256:
  `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`
- Tokenizer class: `Qwen2TokenizerFast`
- `transformers`: `4.57.1`
- Tokenizer lock SHA-256:
  `4c4fa5083c4b6da6097657bb15a304df06080a5692033c1f444288574acdf0b6`

The frozen official file hash matched the server snapshot. Chinese, English, mixed-language,
full-width, emoji, and special-character tokenization probes were recorded in the tokenizer lock.

## Formal donor-map result

| Quantity | Result |
|---|---:|
| Source references | 3,934 |
| Active references | 3,904 |
| Inactive references | 30 |
| Active coverage | 99.2374% |
| Strata | 95 |
| Strata with deactivation | 14 |
| Mean absolute token-length difference | 1.3934 |
| Maximum absolute token-length difference | 67 |

Low-score sample coverage:

| Label | Reason-covered samples | Active samples | Fully inactive |
|---:|---:|---:|---:|
| 1 | 21 | 18 | 3 |
| 2 | 40 | 34 | 6 |

The formal map passed all strict checks: unique donors, no fixed points, no same-sample donors, no
same-content donors, exact stratum preservation, active-set equality, and cycle IDs for every
active reference.

## R2/R3 mask lock

- R2 active-mask SHA-256:
  `9ec8686a4f92d11333de6081c8464c2afb7287aa7b59c3d712998deff11fb2da`
- R3 active-mask SHA-256:
  `9ec8686a4f92d11333de6081c8464c2afb7287aa7b59c3d712998deff11fb2da`
- Byte-identical: `true`

Final status: `R2_STRICT_DONOR_MAP_READY`.

This status allows construction of the frozen S0/R1/R2/R3 training manifests. It does not by
itself authorize smoke or formal training.

## Downstream manifest-schedule compatibility

The reference-level donor map remains valid under the frozen `PASS_STRICT` contract.
However, a downstream audit found that combining this map with the preregistered
one-reference-per-row rotating schedule can change the realized per-rationale frequency
between R2 and R3 when a donor edge crosses rows with different reference counts.

Therefore, this checkpoint still authorizes use of the donor-map artifact as a reviewed
reference-level result, but it does **not** authorize freezing training manifests until the
schedule-frequency issue in `MANIFEST_SCHEDULE_REVIEW.md` is resolved and reviewed.
