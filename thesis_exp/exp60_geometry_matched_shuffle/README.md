# Exp60: geometry-matched shuffled orthogonal control

Exp60 is a draft, train/dev-only mechanism experiment.  It asks whether the
non-collinear residual effect found in Exp59 requires the observed rater
distribution to remain aligned with its own sample.

The three concurrent arms are:

- `consensus_only`: `G_C`;
- `aligned_orthogonal_only`: `G_C + O_A`;
- `matched_shuffled_orthogonal_only`: `G_C + O_pi_tilde`.

`O_pi_tilde` is projected against the same current common backbone gradient
and scaled once to the aligned orthogonal norm.  This makes the component norm,
hypothetical pre-clip total norm and standard clipping coefficient locally
equal.  The claim is local to each arm's current state; future optimization
trajectories are not claimed to remain matched.

All arms retain the same full hard-head hard CE and soft-head empirical soft CE
updates.  Both diagnostic residual VJPs are computed in every arm with restored
RNG state.  Only the residual component injected into the shared backbone
changes.

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
python -m unittest thesis_exp.tests.test_exp60_geometry_matched_shuffle
```

No command above loads a model, takes an optimizer step or accesses test data.
