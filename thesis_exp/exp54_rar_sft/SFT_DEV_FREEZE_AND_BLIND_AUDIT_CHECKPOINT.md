# Exp54 SFT dev freeze and rationale-audit checkpoint

## Completed

| Item | Frozen result |
|---|---|
| Reviewed dev-result commit | `a0c987b6973c5e03130fdee009df347b35d48080` |
| Review verdict | `SFT_DEV_RESULT_PASS` |
| Selected checkpoints | S0/R1/R2/R3 × seeds 42/43/44 |
| Selected epoch | Logical epoch 3 for all 12 checkpoints |
| Selected checkpoint count | 12 |
| Selected dev prediction rows | 7,968 |
| Strict parse | 100% in every selected run |
| Checkpoint-selection lock | `364a2c9ee023d1b0a2d803530d7898a347a150dd89dbaaa388aa44677cd71df4` |
| Independent freeze audit | `SFT_DEV_CHECKPOINT_SELECTION_FREEZE_AUDIT_PASS` |
| Audit-report SHA-256 | `d510a7c94b68808b77fbee24d8bebee3ab46767915926f0e3f6f304404e9a9f2` |
| Test accessed | No |

The freeze binds every selected adapter and trainer state, each selected
metrics/protocol/run-state/predictions file, the complete 36-run dev artifact
tree, the public dev summaries, the vLLM inference implementation, and the
upstream manifest/training/tokenizer/rubric locks.

## Blind-audit preregistration

| Design item | Frozen rule |
|---|---|
| Primary comparison | R3 versus R2 |
| Secondary comparison | R3 versus R1 |
| Sample | 40 unique dev rows, shared by all arms and seeds |
| Low-score coverage | Include every available Label-1/2 dev row |
| Remaining sample | Deterministically select 8 Label-3 and 12 Label-4/5 rows while balancing metric and language |
| Seed replication | Evaluate the same 40 rows for seeds 42, 43, and 44 |
| Pairing | Same record and seed on both sides |
| Order control | Evaluate A/B and swapped B/A; inconsistent results become ties |
| Evaluators | Two independent model families; exact identities still must be frozen |
| Primary forced-close handling | Keep every sampled output; no post-hoc removal |
| Forced-close secondary analysis | Neither, R3 only, comparator only, both |
| Statistical unit | Record-level paired hierarchical bootstrap carrying all three seeds |
| Public claim name | Model-based rationale agreement/preference |
| Forbidden interpretation | Human/expert correctness or faithful hidden reasoning |

## Current boundary and next step

This checkpoint freezes the SFT choice and preregisters the scientific
comparison. It does not authorize evaluator calls, test access, decoder
changes, checkpoint reselection, or preference training.

The next implementation step is to build and hash the deterministic private
40-row sample, choose two eligible evaluator model families, and freeze their
exact revisions, prompts, schemas, decoding settings, and presentation order
in a candidate execution package. That package requires one narrow review
before evaluator calls begin.
