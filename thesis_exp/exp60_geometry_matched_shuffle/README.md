# Exp60: geometry-matched shuffled orthogonal control

Exp60 is a draft, train/dev-only mechanism experiment.  It asks whether the
non-collinear residual effect found in Exp59 requires sample--target alignment
of the observed adjacent 2:1 ordinal residual.

The three formal arms are:

- `consensus_only`: `G_C`;
- `aligned_orthogonal_only`: `G_C + O_A`;
- `matched_shuffled_orthogonal_only`: `G_C + O_pi_tilde`.

`O_pi_tilde` is projected against the same current common backbone gradient
and scaled once to the aligned orthogonal norm.  This makes the component norm,
hypothetical pre-clip total norm and standard clipping coefficient locally
equal.  The claim is local to each arm's current state; future optimization
trajectories are not claimed to remain matched.

All arms use identical hard- and soft-head objectives and identical pre-clip
parameter-support rules.  Their actual gradients may diverge with model state.
The aligned and matched-shuffled arms are locally clip-matched; Consensus-only
is not clip-matched.  Both diagnostic residual VJPs are computed in every arm
with restored RNG state.

The admissible claim is narrow: a positive result supports sample--target
alignment specificity relative to this one preregistered maximum-mismatch
orthogonal control.  It does not identify general human ambiguity, all possible
orthogonal regularizers, or a low-score-specific mechanism.

Formal training is prohibited while
`configs/exp60_geometry_matched_shuffle/protocol_draft.json` has draft status.
A real-Qwen BF16 no-update preflight, independent review, frozen protocol and
source lock are required before the nine formal runs.

CPU preparation commands:

```bash
python -m thesis_exp.exp60_geometry_matched_shuffle.mapping
python -m thesis_exp.exp60_geometry_matched_shuffle.preflight
python -m thesis_exp.exp60_geometry_matched_shuffle.trainer_static_audit
python -m thesis_exp.exp60_geometry_matched_shuffle.source_closure_audit
python -m thesis_exp.exp60_geometry_matched_shuffle.freeze_preflight_contract
python -m unittest thesis_exp.tests.test_exp60_geometry_matched_shuffle
```

No command above loads a model, takes an optimizer step or accesses test data.
The generated `preflight_source_lock.json` has no training authority.  It binds
the reviewed preflight implementation and a normalized scientific-protocol
hash; after preflight, only protocol status, the three observed physical GPU
bindings and an explicit freeze timestamp may change.
After this CPU package passes independent review, the separate real-model
entrypoint may be run for seeds 47--49; it loads Qwen in BF16 but never creates
an optimizer or updates a parameter:

```bash
python -m thesis_exp.exp60_geometry_matched_shuffle.real_model_preflight \
  --seed 47 --gpu_slot 0 --local_files_only
```

Seeds 47/48/49 are fixed to preflight slots 0/1/2.  The finalizer requires
three distinct `CUDA_VISIBLE_DEVICES` values and physical GPU identities.
Every window must have finite, nondegenerate treatment components; each seed
must have at least one window with storage cosine at most 0.99 and relative
distance at least 0.1.

Passing one seed is not sufficient to freeze the formal protocol.  Both 32- and
24-microbatch windows, all three initializations, treatment separation, exact
diagnostic-gradient non-pollution, explicit-route equivalence and the memory
margin must pass first.
