# Exp54 V2 revision checkpoint

This checkpoint records the V2 decoder fixes and the later decision to return
dev execution to the normal operator-controlled research workflow. It does
not change the grammar, XGrammar token mask, UTF-8 incomplete-prefix
completion, 256-token budget completion, model checkpoints, training data,
or training loss.

## Closed blockers

1. The authorization returns one authenticated execution context containing
   the exact read-once protocol, grammar, training configuration, sealed dev,
   and checkpoint payloads. The runner consumes those payloads directly and
   verifies all identities and hashes again before imports/model loading,
   after adapter loading, and before writing predictions.
2. The trust-anchor, staged-digest and claim-directory mechanism has been
   retired from the active runner. The direct runner reserves each ordinary
   deterministic result directory with `exist_ok=False` before dev or model
   access, so accidental duplicate launches fail without requiring special
   permissions, root ownership, append-only flags or `sudo`.
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
  `671b47ec46f2c5ec8e178001606d6047855e22fc107f8c553a268176e00db4ae`
- Candidate lock SHA-256:
  `96a44a387b997b129129ded267ffa5cca8523e18b5d44f9cd8f7a0e5952d560d`

All evidence generation remained train-only or synthetic:
`v2_dev_accessed=false`, `test_accessed=false`, and
`formal_test_allowed=false`. The regenerated report records
`operator_direct_dev_execution_allowed=true`; this is an execution decision,
not a change to the decoder or test-access boundary.

## Execution boundary

The exact direct runner may be synchronized to the server and launched after
the operator confirms an idle allowed GPU. This checkpoint does not
authorize test access. Existing result directories and logs must be retained;
Codex must not silently delete or rerun a failed task.
