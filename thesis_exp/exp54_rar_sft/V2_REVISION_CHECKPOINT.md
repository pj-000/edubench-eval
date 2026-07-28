# Exp54 V2 revision checkpoint

This checkpoint closes only the four blockers from the prior V2 review. It
does not change the grammar, XGrammar token mask, UTF-8 incomplete-prefix
completion, 256-token budget completion, model checkpoints, training data,
or training loss.

## Closed blockers

1. The authorization returns one authenticated execution context containing
   the exact read-once protocol, grammar, training configuration, sealed dev,
   and checkpoint payloads. The runner consumes those payloads directly and
   verifies all identities and hashes again before imports/model loading,
   after adapter loading, and before writing predictions.
2. One-use claims now live under the fixed root-managed
   `/var/lib/edubench/exp54-v2-dev/<campaign>` hierarchy. Preflight requires
   root ownership, a training-group campaign directory, exact modes, and the
   append-only filesystem flag. Deleting an output cannot restore invocation
   permission; a failed claimed task requires a new reviewed campaign.
   The campaign mode is `01770`, so the training group can execute the real
   preflight directory read while append-only protection still prevents claim
   deletion or rename.
3. The exact XGrammar and apache-tvm-ffi wheels are unpacked in memory and
   compared byte-for-byte with their installed distributions, excluding only
   explicit installer-generated metadata, bytecode, and console scripts.
   Source-directory and VCS installs are rejected.
4. The independent candidate auditor hard-binds the upstream materialized
   manifest and reference-set lock digests, locked train hash, frozen shared
   prompt cache, and all 12 manifest hashes. It requires each score target to
   equal the unique locked train label and derives language only from that
   unique locked train record because language is not duplicated in the
   frozen manifest schema.
5. Production authorization now authenticates the sealed-dev hash without
   materializing rows, wins the one-use claim, reserves the fixed output, and
   only then parses rows from the already authenticated payload. Concurrent
   losers never receive materialized dev rows; parse failure retains the
   claim.

## Server evidence

- CPU tests: 70 passed, 0 failed, 0 skipped.
- Train-only structural smoke: 36/36 strict parses.
- Forced completion: 13/36 (36.11%), retained as a diagnostic.
- Locked budget probe: completion at token 256, strict parse pass.
- Real checkpoint determinism: output bytes, token IDs, and diagnostics all
  identical across two runs.
- Runtime source closure: 26/26.
- Train-only smoke report SHA-256:
  `63ccd3db04e7a020d857f08f637d9edd396103b3294b0a42783e9199876af2f6`
- Candidate report SHA-256:
  `3bd4000766e0fe96143e2ce9f12e93395686431189fcfcbe0b5a9d6830d4b08f`
- Candidate lock SHA-256:
  `97ea2c454a791a763cbe3c8f7e05b1cfe0bf6ee162d2bfc2af5aed6357734624`

All evidence generation remained train-only or synthetic:
`v2_dev_accessed=false`, `test_accessed=false`,
`formal_v2_dev_allowed=false`, and `formal_test_allowed=false`.

## Authorization boundary

The current commit is still a candidate and cannot run formal dev by itself.
After independent review passes, a separate authorization payload must follow
the staged pre-activation process in `V2_DEV_AUTHORIZATION_PLAN.md`. This
checkpoint does not install an authorization, does not create a formal claim,
and does not authorize test access.
