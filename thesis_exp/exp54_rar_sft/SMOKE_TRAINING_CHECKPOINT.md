# Exp54 deterministic train-only smoke checkpoint

## Status

The reviewed training configuration at commit
`d2762a427b8bed957459b04ba239a988ab3acea5` is frozen. A deterministic
train-only smoke subset and a separate diagnostic runner now exist, but no
smoke or formal execution is authorized.

The one-use smoke guard at commit
`b2e616e324ed9de7c8aaf8eb1dee5a2f0d7a53bb` received
`SMOKE_GUARD_PASS`. A deterministic execution-authorization candidate and
root-managed installation plan now exist for the next review, but neither the
authorization nor its trust anchor has been installed.

Current hard state:

- trust-anchor installation: forbidden
- model loading for smoke: forbidden without a future external authorization
- forward/backward: forbidden
- smoke training: forbidden
- formal training: forbidden
- dev accessed: false
- test accessed: false
- training used: false

## Configuration freeze

`training_configuration_frozen_lock.json` binds:

- verdict `TRAINING_CONFIGURATION_PASS`;
- exact reviewed commit
  `d2762a427b8bed957459b04ba239a988ab3acea5`;
- candidate report SHA-256
  `27186421a67aa5c0b324672e567280859295e3066201fbca5b97ae2cd07dc0aa`;
- candidate lock SHA-256
  `a69b7c4594d86d8dbffa432550d0d69a0d39bd67785c89dc8045abf676e4ee0a`;
- the exact training configuration and materialized-manifest freeze;
- the reviewed 16-file formal runtime-source closure.

The configuration-freeze lock SHA-256 is
`85c098fda8af2d852874107249dbf89c03b0b59424fad8336096ffdaa37d31f7`.

## Deterministic smoke selection

The private smoke package uses only seed 42, logical epoch index 0 from the
already-frozen train manifests. It contains eight events per arm.

The eight predeclared selection slots are:

1. rationale active, score 1
2. rationale active, score 2
3. rationale active, score 3
4. rationale active, score 5
5. rationale inactive, score 1
6. rationale inactive, score 2
7. rationale inactive, score 4
8. rationale inactive, score 5

For each slot, the builder selects the event with the minimum SHA-256 of the
canonical tuple:

```text
[
  "exp54-smoke-selector-v1",
  42,
  0,
  base_event_id
]
```

The final eight events are ordered by that digest. No row was manually chosen.
The selected event vector and order are identical across S0/R1/R2/R3.

The selected score histogram is:

- score 1: 2
- score 2: 2
- score 3: 1
- score 4: 1
- score 5: 2

Rationale-active counts are:

- S0: 0
- R1: 4
- R2: 4
- R3: 4

The four arms each contain eight events and exactly 16,384 fixed-padded
tokens. They use micro-batch size 2 and gradient accumulation 4, producing
exactly one optimizer step per independently initialized arm.

This is a pipeline diagnostic, not a miniature effect estimate. R1/R2/R3 may
have different unpadded and rationale-supervised token totals inside these
eight events. Smoke losses must not be compared to choose an arm,
hyperparameter, checkpoint, or scientific conclusion.

## Private and public artifacts

Private files remain under the ignored server-only
`rar_v2/data/smoke_v1_stratified/` directory:

- one eight-row manifest for each arm;
- one eight-row prompt cache.

The public report contains only aggregate counts and cryptographic hashes. It
does not publish record IDs, reference IDs, event IDs, token-ID rows, donor
edges, or human rationale text.

Public hashes:

- smoke package report:
  `35ab0cfed0e4e6745887ef46b605789e8ddcb2f39c8dfef21c0db27fd875acad`
- smoke package frozen lock:
  `90509b87d19ec93da6e0ebc51560cf500ae94181ce1c1a64c36d24001785b150`

The independent auditor reconstructs all eight selections from the frozen full
manifests without importing the production builder, then verifies every
private row, prompt row, public hash, source hash, runtime closure, and
authorization boundary.

Authorization-candidate hashes:

- authorization JSON:
  `88918747e68edb0fe838de197a8ac41d0b2fd1d9b275697855c30c6db23d8c97`
- one-line digest candidate:
  `2665ab0ce22730c43ab2f34f739d0ee087f1bdc633237b4702e73bc87652d30e`
- authorization candidate report:
  `f7f74a541b12ce7648549d59ef06b93254bc15509f6edcb8d72adcfa755a7640`

## Future smoke authorization

The smoke runner uses a separate trust anchor:

```text
/etc/edubench/exp54_smoke_authorization.sha256
```

The formal-training trust anchor is not reused. A future
`SMOKE_PACKAGE_PASS` authorization must bind:

- the exact reviewed smoke-package commit
  `e3c642abca96f3f88034caecfae30fda596c2827`;
- the training-configuration frozen-lock SHA-256;
- the smoke-package frozen-lock SHA-256;
- the smoke-plan SHA-256;
- the materialized-manifest frozen-lock SHA-256;
- the 20-file smoke runtime-source closure SHA-256;
- exactly arms S0/R1/R2/R3;
- exactly seed 42;
- exactly one optimizer step per arm;
- one campaign ID, one unique run ID and one fixed output directory per arm;
- `max_invocations_per_arm=1`;
- claim root `/var/lib/edubench/exp54-smoke`;
- `formal_training_allowed=false`;
- `dev_accessed=false`;
- `test_accessed=false`;
- `hyperparameter_selection_allowed=false`.

The exact authorization-file SHA-256 must then be installed externally as a
root-owned, non-symlink, non-group/other-writable file. Until that happens,
the runner hard-fails before model verification/loading, CUDA initialization,
output creation, forward, or backward.

The guard reads the authorization, configuration, plan, smoke lock, and
configuration lock once each; every digest and parsed object comes from the
same byte payload. Private manifests and the prompt cache follow the same
read-once rule. After authorization, the runner atomically creates one
non-reusable `<arm>.claimed` file in the root-managed append-only campaign
directory, then atomically reserves the authorization-bound output directory.
Claim failure occurs before model verification or CUDA initialization.

## Acceptance after a future authorized run

Each arm must independently satisfy:

- eight events and four micro-batches;
- exactly one optimizer step;
- finite total, score, and active-rationale losses;
- finite positive pre-clipping gradient norm;
- score supervision on every event;
- rationale masks exactly matching activity;
- adapter-only model checkpoint whose safetensors keys, shapes, dtypes, and
  parameter count exactly match the in-memory PEFT LoRA state;
- no extra safetensors, binary/index weight file, symlink, or base-model
  tensor in the output tree;
- no dev/test access;
- result marked diagnostic-only and unusable for model selection.

Passing smoke authorizes neither seed-42 formal training nor any other formal
run. Formal execution requires a later, separate authorization gate.
