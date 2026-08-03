# Exp60 independent-review package

## Decision status

`READY_FOR_INDEPENDENT_REVIEW_NOT_AUTHORIZED_FOR_FORMAL_TRAINING`

No model has been loaded, no optimizer step has been taken, no GPU has been
used, and test remains inaccessible.  The protocol deliberately remains a
draft.  There is no Exp60 source lock yet.

## Scientific question

Does the non-collinear residual benefit found in Exp59 require the observed
rater distribution to remain aligned with its own sample, or can an equally
large mismatched orthogonal residual reproduce it?

## Draft intervention

- `consensus_only`: `G_C`;
- `aligned_orthogonal_only`: `G_C + O_A`;
- `matched_shuffled_orthogonal_only`: `G_C + O_pi_tilde`.

All three arms execute the same ordinary aligned hard/soft objective and both
diagnostic residual VJPs with restored RNG.  Hard- and soft-head updates remain
the complete original updates.  Only the residual component injected into the
shared backbone differs.

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

## Completed pre-result checks

- Nine NumPy mapping/geometry/analysis tests pass.
- Four PyTorch CPU tests of the memory-streaming formal geometry step pass.
- Exp59 regression suite: seven tests pass.
- Restored historical Exp51 suite: eight pass, one Torch-dependent collection
  is skipped in the default environment.
- Exp49--Exp51 source/config closure: 43/43 blobs exactly match commit
  `fa72bd4`.
- Trainer static audit confirms two diagnostic VJPs, one standard clip and one
  optimizer step.
- Generic 64-case CPU preflight maxima:
  - aligned orthogonality error: `1.84e-08`;
  - shuffled orthogonality error: `1.31e-08`;
  - component-norm match error: `5.95e-08`;
  - pre-clip total-norm match error: `3.67e-08`;
  - clip-coefficient match error: `3.67e-08`.

These are generic FP32/FP64 checks.  A real Qwen BF16 no-update preflight is
still mandatory before formal training.

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

## Questions requiring independent approval

1. Does the maximum-mismatch mapping isolate sample alignment adequately while
   preserving the intended residual multiset?
2. Is matching the shuffled component to the aligned norm at each arm's own
   current state the correct feasible estimand?
3. Are fixed epoch 10, fresh seeds 47--49, and the proposed practical gates
   adequate for the one-shot decision?
4. Is keeping the empirical aligned soft-head CE identical in all arms the
   right head-update control?
5. Does the preregistered item-weighted question-cluster bootstrap correctly
   state that it is conditional on the three trained seeds?
6. What additional real-model no-update checks are mandatory before freezing?

Formal training must remain blocked until these questions are reviewed, the
real-model preflight passes, the protocol status is changed exactly once, and a
source lock is generated.
