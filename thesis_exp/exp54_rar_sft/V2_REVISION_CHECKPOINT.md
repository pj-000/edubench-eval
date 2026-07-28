# Exp54 V2 revision checkpoint

This checkpoint closes only the three blockers from the prior V2 review. It
does not change the grammar, XGrammar token mask, UTF-8 incomplete-prefix
completion, 256-token budget completion, model checkpoints, training data,
or training loss.

## Closed blockers

1. The formal runner now requires a fixed-path external authorization as its
   first operation. The authorization binds the reviewed commit and candidate
   artifacts, the sealed dev hash, the complete source/runtime/dependency
   identity, all 36 checkpoint artifacts, batch size 16, token budget 256,
   fixed outputs, `test_allowed=false`, and one invocation per checkpoint.
2. The formal runtime closure is an explicit 26-file set covering every local
   prompt, normalization, manifest, model-validation, parser, decoder,
   launcher, package-initializer, and configuration dependency. Actual wheel
   bytes and installed distribution file trees are independently recorded;
   editable installs are rejected.
3. The candidate auditor independently reconstructs the 36 train-only smoke
   selections from frozen train/manifests/prompt cache. It does not import the
   production selector or production JSON/hash helpers.
4. Formal batch size is frozen at 16 with no CLI override. Mixed prompt length
   and mixed completion-time regression proves batched and singleton token
   IDs, output bytes, strict parse results, and diagnostics are identical,
   while completed rows receive attention mask 0 and unfinished rows continue
   through the KV cache.

## Server evidence

- CPU tests: 46 passed, 0 failed, 0 skipped.
- Train-only structural smoke: 36/36 strict parses.
- Forced completion: 13/36 (36.11%), retained as a diagnostic.
- Locked budget probe: completion at token 256, strict parse pass.
- Real checkpoint determinism: output bytes, token IDs, and diagnostics all
  identical across two runs.
- Runtime source closure: 26/26.
- Candidate report SHA-256:
  `6417661348e538f16977833409c931168c09009eef60cbbbd27244ddf3e5af5a`
- Candidate lock SHA-256:
  `185949d40c50f2d2363ec7c2facaf2dcace437ef84ced14bafbc2f9e55a432a1`

All evidence generation remained train-only or synthetic:
`v2_dev_accessed=false`, `test_accessed=false`,
`formal_v2_dev_allowed=false`, and `formal_test_allowed=false`.

## Authorization boundary

The current commit is still a candidate and cannot run formal dev by itself.
After independent review passes, a separate authorization payload must follow
the staged pre-activation process in `V2_DEV_AUTHORIZATION_PLAN.md`. This
checkpoint does not install an authorization, does not create a formal claim,
and does not authorize test access.
