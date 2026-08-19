# SORC-DPO deterministic train-only smoke package

Status: candidate smoke package and runner implementation only. GPU smoke,
formal preference training, evaluator calls, dev, and test remain forbidden.

## Reviewed training freeze

`PREFERENCE_TRAINING_IMPLEMENTATION_PASS` is bound to exact commit:

```text
82a22f842b2f5009962ba0019bcae61f172d5797
```

The reviewed loss, collator, materialized manifests, and training config are
promoted by `preference_training_frozen_lock.json`. Any bound public/private
artifact or source hash change invalidates that freeze.

## Deterministic train-only subset

Selection uses ascending:

```text
SHA256("exp54-sorc-dpo-smoke-v1|record_id|pair_type")
```

No dev/test row or evaluator output participates.

| Arm | Score composition | Rationale pairs | Total pairs | Smoke updates |
|---|---|---:|---:|---:|
| P1 Field-DPO | 16 adjacent + 8 L2H + 8 H2L guard | 0 | 32 | 1 |
| P2 SORC-score | same 32 records/blocks as P1 | 0 | 32 | 1 |
| P3 Joint SORC | same 32 score records/blocks | 59 | 91 | 1 |
| P1-SYN | same 32 records/blocks as P1 | 0 | 32 | 1 |

P1, P2, and P1-SYN share the exact record/block vector. P1 and P1-SYN also
share chosen sequence and chosen field-mask vectors. P3 reuses the P2 smoke
score vector before its independently selected rationale rows.

## Runner boundary

`train_sorc_dpo_smoke.py` supports two mutually exclusive modes:

- `--validate-only`: CPU validation and collation only;
- `--execute`: requires a separate exact GPU-smoke authorization file.

The repository does not contain that authorization. It must bind:

- exact arm and seed 42;
- exactly one optimizer step;
- exactly one physical pair per micro-batch;
- smoke package, smoke plan, preference-training config, and runner hashes;
- the exact frozen R3 checkpoint-selection lock;
- the exact base-training configuration and canonical base-model snapshot
  identity;
- the exact frozen preference-training lock and smoke-implementation lock;
- one visible `NVIDIA RTX A6000`, its immutable CUDA UUID, and a
  pre-registered free-memory floor of at least 40 GiB before model loading;
- one fixed absolute output directory;
- formal-training/dev/test flags all false.

The checkpoint lock is no longer a CLI override. Its resolved seed-42 R3
adapter directory must remain below the fixed `formal_runs` root, so an
absolute path or `..` traversal is rejected.

P3 additionally requires a final rationale-qualification lock with the exact
frozen field `p3_preference_training_allowed=true`,
`rationale_blind_qualification_completed=true`, two completed evaluator-family
decisions, frozen source hashes, and dev/test both false. The deprecated
`p3_training_allowed` spelling is rejected rather than treated as an alias.
Missing or mismatched authorization fails before torch, CUDA, PEFT,
Transformers, model loading, or forward/backward.

The runner currently preserves the reviewed physical micro-batch of one pair.
No A6000 performance/memory profile has been selected or run. The first GPU
gate is a correctness smoke at physical batch one, not a throughput search.
Only after all four one-step correctness results pass may a separate,
train-only P3 capacity probe test fresh-process candidates `1, 2, 4, 8, 16`.
That later probe must retain the real 91-pair group denominator, use only OOM
and peak reserved memory (never semantic outputs) for selection, and choose the
largest batch with peak reserved memory no greater than 46 GiB. It requires a
separately reviewed batch-general runner; the current runner cannot silently
enable it.

## Current validation

- Four arm-specific CPU validation runs: PASS.
- Unauthorized `--execute`: hard-fails on the missing authorization file.
- Independent deterministic subset/public budget audit: PASS.
- GPU initialized: false.
- Model loaded: false.
- Forward/backward executed: false.
- Dev/test accessed: false.

The next important gate is not another review of DPO mathematics. It is a
narrow GPU-smoke authorization decision covering runner correctness, the
one-step outputs, and the A6000 execution/memory profile.
