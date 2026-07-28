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

## Server evidence

- CPU tests: 62 passed, 0 failed, 0 skipped.
- Train-only structural smoke: 36/36 strict parses.
- Forced completion: 13/36 (36.11%), retained as a diagnostic.
- Locked budget probe: completion at token 256, strict parse pass.
- Real checkpoint determinism: output bytes, token IDs, and diagnostics all
  identical across two runs.
- Runtime source closure: 26/26.
- Train-only smoke report SHA-256:
  `63ccd3db04e7a020d857f08f637d9edd396103b3294b0a42783e9199876af2f6`
- Candidate report SHA-256:
  `85b7d9656afee6b1d9772435cd41eda5ff75715b84883d185ae033f7c0d30c92`
- Candidate lock SHA-256:
  `6c8bcb196a204134b8ecb4231b0441488b2a82caaa4f6cd0d47cf3efe788eff1`

All evidence generation remained train-only or synthetic:
`v2_dev_accessed=false`, `test_accessed=false`,
`formal_v2_dev_allowed=false`, and `formal_test_allowed=false`.

## Authorization boundary

The current commit is still a candidate and cannot run formal dev by itself.
After independent review passes, a separate authorization payload must follow
the staged pre-activation process in `V2_DEV_AUTHORIZATION_PLAN.md`. This
checkpoint does not install an authorization, does not create a formal claim,
and does not authorize test access.
