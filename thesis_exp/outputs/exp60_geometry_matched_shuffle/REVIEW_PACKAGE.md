# Exp60 independent-review package

## Decision status

`READY_FOR_INDEPENDENT_REVIEW_NOT_AUTHORIZED_FOR_FORMAL_TRAINING`

No optimizer step has been taken and test remains inaccessible.  A real-model
no-update qualification attempt exposed a BF16 numerical-gate mismatch and
stopped before training; it produced no authorized seed report.  The protocol
remains a draft and there is no formal Exp60 source lock.  A separate no-
training-authority preflight source lock binds the CPU/GPU-preflight code.

- preflight source-lock SHA-256:
  `b8f526856fa1de35ff89aa64863518a57607bd81982d6ea8dde402b74376cf76`;
- normalized scientific-protocol SHA-256:
  `2efa551e3d2ea8e9e0be94c0c800696add1d569698d970d9967f5102af5391c5`.

## Scientific question

Does the non-collinear residual benefit found in Exp59 require sample--target
alignment of the observed adjacent 2:1 ordinal residual, or can this one
preregistered geometry-matched maximum-mismatch residual reproduce it?

## Draft intervention

- `consensus_only`: `G_C`;
- `aligned_orthogonal_only`: `G_C + O_A`;
- `matched_shuffled_orthogonal_only`: `G_C + O_pi_tilde`.

All three arms use identical hard- and soft-head objectives and identical
pre-clip parameter-support rules.  Actual gradients may diverge with model
state.  Aligned and matched-shuffled are locally clip-matched; Consensus-only
is not clip-matched.  Both diagnostic VJPs use restored RNG states.

The shuffled component is independently projected against the current common
backbone gradient and scaled once to the aligned orthogonal norm.  Consequently
the component norm, hypothetical pre-clip total norm and standard clipping
coefficient are locally matched.  This is an instantaneous intervention within
each arm's current state; it does not claim that future trajectories remain
matched after the arms diverge.

## Mapping audit

The deterministic within-hard-label rotation preserves every target-vector
multiset and reaches the categorical maximum possible mismatch:

- rows: 2,654;
- effectively changed: 2,512;
- effective change rate: 94.6496%;
- self assignments: 0;
- mapping SHA-256: `dc9d3fbce24af01fee58e896f83ed35f3ccd516fcf30f35ed43a3cfd4c2fa7fb`.

The 142 unavoidable unchanged targets arise from dominant target categories in
hard-label strata; changing them is impossible under an exact multiset-
preserving bijection.

The regenerated audit additionally verifies that donor and recipient ID sets
are identical, every donor is used exactly once, hard labels match, and every
stored shuffled target equals the donor's source target.  Label 1 changes only
8/24 rows, so Exp60 cannot support a low-score-specific mechanism claim.

## Claim boundary

A positive result can support only sample--target alignment specificity of the
observed ordinal residual relative to this canonical control.  It cannot by
itself establish general human ambiguity, multimodal preferences, subgroup
structure, all possible orthogonal regularizers, or low-score mechanisms.

## Completed pre-result checks

- Eighteen generic mapping/geometry/config/contract/analysis tests pass.
- Seven PyTorch CPU tests, including BF16 storage-space simulation, non-finite
  rejection and zero-treatment rejection, pass.
- Exp59 regression suite: seven tests pass.
- Restored historical Exp51 suite: eight pass, one Torch-dependent collection
  is skipped in the default environment.
- Exp49--Exp51 source/config closure: 43/43 blobs exactly match commit
  `fa72bd4`.
- Trainer static audit confirms two diagnostic VJPs, one standard clip and one
  optimizer step in formal training; formal source-lock verification is now
  unconditional and all frozen runtime settings are code-checked.
- Generic 64-case CPU preflight maxima:
  - aligned orthogonality error: `1.84e-08`;
  - shuffled orthogonality error: `1.31e-08`;
  - component-norm match error: `5.95e-08`;
  - pre-clip total-norm match error: `3.67e-08`;
  - clip-coefficient match error: `3.67e-08`.

These are generic FP32/FP64 checks.  A real Qwen BF16 no-update preflight is
still mandatory before formal training.

## Revisions after independent review

- The formal trainer can no longer bypass source lock through an environment
  variable and also requires the aggregated real-model-preflight decision.
- Model path, epochs, optimizer settings, batch sizes, accumulation, clipping,
  length, BF16, checkpointing, no-subsampling, workers, data hashes and mapping
  hash are machine-checked.
- Both aligned and shuffled candidates are audited after storage-dtype casting
  for orthogonality, component norm, total norm and clipping coefficient.
- BF16 storage-space normalized orthogonality uses a separate `0.10` numerical
  qualification bound (an angle of at least about 84.3 degrees).  This was
  amended after no-update train-window qualification showed that the original
  unit-roundoff-derived `0.01171875` bound was inapplicable to a small residual
  obtained by subtracting two large quantized gradients.  Norm, total-norm and
  clip matching remain at `0.01171875`; no dev metric or optimizer result was
  observed when making this numerical amendment.
- The no-update qualification also calibrates explicit-route subtraction error
  to `0.35`: independent BF16 accumulated routes differ by 0.29--0.33 relative
  to the much smaller residual, while head errors remain exactly zero, forward
  logits match and diagnostic capture leaves the main gradient unchanged.
- Conservative projected formal memory must retain at least 8% free capacity;
  all three 3090 observations retain 9.1--9.9% (about 2.3 GB) after adding a
  full FP32 Adam-state estimate.  This gate does not use dev or test outcomes.
- Aligned--shuffled cosine and relative distance are recorded.
- A separate real-Qwen BF16 no-update preflight now covers 32- and
  24-microbatch windows, exact diagnostic-gradient non-pollution, RNG/logit
  parity, explicit-route residual equivalence, memory and runtime.  It contains
  no optimizer.
- Formal variants rotate through three GPU slots by a frozen Latin square.
- Analysis recomputes metrics from all 664 unique frozen dev predictions and
  verifies IDs, targets, hashes, initializations, batch order, source lock and
  all geometry gates.
- The aligned-vs-consensus secondary requirement is now at least 2/3 favorable
  seeds and mean MAE delta at most -0.005.  Adding seeds after results is
  prohibited.

## Revisions after review of commit 72d266a

- Every gradient, residual, candidate vector, reduction, projection scalar,
  cosine, relative error and clip result now fails closed on NaN/Inf;
  `clip_grad_norm_` uses `error_if_nonfinite=True`.
- Storage-space treatment activity must be at least `1e-6`; zero/near-zero
  components cannot be reported as a separated direction.
- Separation is evaluated per seed: both windows must be finite and
  nondegenerate, and at least one must jointly satisfy cosine `<=0.99` and
  relative distance `>=0.1`.
- Real preflight directly rehashes the actual mapping JSONL and verifies exact
  coverage of all 2,654 frozen train IDs.
- `preflight_source_lock.json` locks the preflight implementation and records a
  normalized scientific-protocol hash.  Final freezing permits changes only to
  status, physical GPU bindings and the explicit freeze timestamp.
- Seeds 47/48/49 must cover slots 0/1/2 and three distinct physical GPU
  identities; final protocol bindings must equal those actually preflighted.
- Formal analysis now revalidates the full source lock, reads all 210 geometry
  rows per endpoint, checks the 32/.../24 window structure, independently
  recomputes extrema, rejects non-finite metrics and verifies physical GPU
  bindings.

## Revisions after review of commit 7b71dc9

- GPU identity is now fail-closed.  A `CUDA_VISIBLE_DEVICES` alias can no
  longer serve as physical identity: PyTorch and/or `nvidia-smi` must yield a
  stable UUID, both sources must agree when present, `nvidia-smi` must provide
  a PCI bus ID, and MIG environments are rejected.
- The all-seed finalizer requires three nonempty, distinct stable GPU UUIDs.
  Formal training resolves the current stable UUID again and compares it with
  the corresponding locked preflight report, in addition to checking the
  visible-device binding.
- The final source lock now has an explicit schema version and mandatory-file
  manifest.  `verify_contract()` independently checks its status, protocol and
  preflight-decision hashes, normalized scientific-protocol hash, frozen
  analysis flag, GPU-binding flag, split contract, file count and mandatory
  subset before verifying every file hash.  Empty or partial manifests fail.
- The post-clip squared norm is now explicitly finite-checked before square
  root.  Formal summaries and analysis additionally audit the complete BF16
  storage-space cosine, relative-distance and per-epoch activity trajectory.
- Model-manifest hashing explicitly excludes only ModelScope's unreadable
  `.msc` download metadata; all model, tokenizer and readable metadata files
  remain hashed.
- Forty-four relevant Exp60/Exp59/Exp51 regression tests pass; both the static
  trainer audit and generic 64-case CPU no-update preflight pass.  No optimizer
  step or test access occurred.

## Files to review

- Draft protocol:
  `thesis_exp/configs/exp60_geometry_matched_shuffle/protocol_draft.json`
- Mapping algorithm:
  `thesis_exp/exp60_geometry_matched_shuffle/mapping.py`
- Geometry definitions and generic preflight:
  `thesis_exp/exp60_geometry_matched_shuffle/geometry.py`
  `thesis_exp/exp60_geometry_matched_shuffle/preflight.py`
- Formal memory-streaming trainer:
  `thesis_exp/exp60_geometry_matched_shuffle/train.py`
- Real-model preflight and three-seed finalizer:
  `thesis_exp/exp60_geometry_matched_shuffle/real_model_preflight.py`
  `thesis_exp/exp60_geometry_matched_shuffle/finalize_real_preflight.py`
- Preflight scientific contract and no-authority lock builder:
  `thesis_exp/exp60_geometry_matched_shuffle/contract.py`
  `thesis_exp/exp60_geometry_matched_shuffle/freeze_preflight_contract.py`
- Post-approval formal source-lock builder:
  `thesis_exp/exp60_geometry_matched_shuffle/freeze_source_lock.py`
- Pre-result analysis implementation:
  `thesis_exp/exp60_geometry_matched_shuffle/analyze_confirmation.py`
- Static and source-closure audits:
  `thesis_exp/exp60_geometry_matched_shuffle/trainer_static_audit.py`
  `thesis_exp/exp60_geometry_matched_shuffle/source_closure_audit.py`
- Tests:
  `thesis_exp/tests/test_exp60_geometry_matched_shuffle.py`
  `thesis_exp/tests/test_exp60_torch_geometry.py`
- Generated evidence:
  `thesis_exp/outputs/exp60_geometry_matched_shuffle/audit/`

## Questions requiring renewed independent approval

1. Does the maximum-mismatch mapping isolate sample alignment adequately while
   preserving the intended residual multiset?
2. Is matching the shuffled component to the aligned norm at each arm's own
   current state the correct feasible estimand?
3. Are fixed epoch 10, fresh seeds 47--49, and the proposed practical gates
   adequate for the one-shot decision?
4. Do the new BF16 storage-space gates, exact no-pollution audit, explicit-route
   equivalence threshold (0.02), treatment-separation rule and memory estimate
   authorize real-model no-update preflight?
5. Does the preregistered item-weighted question-cluster bootstrap correctly
   state that it is conditional on the three trained seeds?
6. Is the strengthened analysis and Latin-square execution contract complete?

Formal training must remain blocked until these questions are reviewed, the
real-model preflight passes, the protocol status is changed exactly once, and a
source lock is generated.
