# Exp64 research specification: optimizer-state-aware residual attribution

Status: **frozen on 2026-08-09 before any independent seed outcome was
observed; five base trajectories and the single formal counterfactual audit
are authorized, while test evaluation remains forbidden**.

## 1. Scientific question

For the signed residual obtained by exactly decomposing a multi-rater soft
cross-entropy target, does the residual's *exact AdamW-attributable parameter
displacement* predict its local hard-task utility on a group-disjoint probe
better than raw Euclidean gradient geometry and simpler optimizer
approximations?

This is a mechanism audit, not a new deployable optimizer and not a claim that
one-step behavior explains all long-run HMSA gains.

## 2. Why this experiment exists

Exp63 compared four candidates after scaling each complete gradient separately
to norm 0.95.  It therefore controlled total raw-gradient norm and clipping,
but it also applied a different scale to the common gradient and both heads in
each arm.  Exp63 is a valid comparison of directions on one raw-gradient norm
sphere; it is not a fixed-common-update residual intervention.

Exp63 also showed that isolated raw-space orthogonal residuals did not pass the
registered seed-level criterion.  This motivates one final test of whether the
actual stateful AdamW displacement is a more informative local coordinate than
raw Euclidean orthogonality.

## 3. Independent confirmation cohort

- Dataset: frozen Exp62 SummEval coherence/fluency train and development data.
- Test split: forbidden; access count remains zero.
- Model: Qwen3-Reranker-0.6B.
- Base route: direct residual blocked (DRB / consensus-only).
- New seeds: 72--76.  These seeds were not used in Exp62 or Exp63 and are
  frozen before any Exp64 model outcome exists.
- Base epochs: 10; complete model, AdamW, scheduler, RNG and next-window state
  saved after epochs 2, 5 and 8.
- AdamW is frozen to `betas=(0.9,0.999)`, `eps=1e-8`, `amsgrad=False`,
  `maximize=False`, `foreach=False`, `fused=False` and `capturable=False` so
  that the tested finite-difference operation order is also the formal
  training operation order.
- Formal counterfactual window: the first 112-record batch of the immediately
  following epoch under a deterministic epoch-specific order.
- No Full/Parallel/Orthogonal long-run trajectories are trained.

Existing Exp63 EduBench states may be used only for implementation checks.  As
they motivated the hypothesis, they cannot be the sole confirmatory evidence.

## 4. Gradient objects

At a frozen state let `G_C` be the complete all-parameter DRB gradient and let
`R` be the analytically isolated signed residual on the trainable backbone;
`R` is exactly zero on both heads.  In the raw backbone parameter space:

```
R = R_parallel + R_orthogonal
<R_orthogonal, G_C_backbone> = 0
```

The five candidates are:

```
blocked               G_C
full_residual         G_C + R
parallel_only         G_C + R_parallel
orthogonal_only       G_C + R_orthogonal
sign_flipped_residual G_C - R
```

## 5. Shared scaling and clipping

One scale is shared by every candidate at a seed-stage state:

```
c_t = min(1, 0.95 / max_arm ||g_arm||_2)
```

Every arm receives `c_t * g_arm`, followed by the common global clipping
threshold 1.0.  Required gates:

- all candidates have pre-clip norm at most 0.95 plus the frozen BF16
  tolerance;
- every effective clip coefficient is exactly 1 within tolerance;
- the common gradient and both heads have exactly the same scale in every arm;
- no per-arm norm matching or per-arm BF16 recalibration is allowed.

This keeps the common update fixed while changing only the residual component.

## 6. Exact AdamW-attributable displacement

For complete frozen optimizer state `S_t`, define the actual one-step parameter
displacement:

```
Delta_S(g) = AdamWStep(theta_t, S_t, g) - theta_t
```

The residual-attributable displacement is the finite difference:

```
delta_R = Delta_S(c_t (G_C + R)) - Delta_S(c_t G_C)
```

Parallel, orthogonal and sign-flipped analogues are computed from independently
restored copies of the same state.  The AdamW non-additivity diagnostic is:

```
iota = delta_R - delta_parallel - delta_orthogonal
```

`iota != 0` only establishes optimizer-map non-additivity.  It is not evidence
that the non-additivity is beneficial.

## 7. Group-disjoint cross-probes

The 15 SummEval development source-article groups are ordered by SHA-256 of
`"exp64_probe_v1\t" + group_id`; alternating groups form probe A (8 groups)
and probe B (7 groups).  The exact group lists and record hashes must be
materialized and source-locked before model training.

For A-to-B:

```
v_A = grad_theta L_hard,A(theta_t)
q_exact = <v_A, delta_R>
Y_R = L_hard,B(theta_after_full) - L_hard,B(theta_after_blocked)
```

Negative `q_exact` predicts a beneficial negative `Y_R`.  B-to-A is computed
symmetrically.  The two directions are averaged within seed after the three
fixed stages; they are not independent replicates.

Using one probe to both construct and validate the Taylor prediction is
prohibited.

The probe gradient and probe loss are accumulated in deterministic batches of
32 records.  This is a memory-only implementation revision made after an
initial batch-128 attempt failed with CUDA OOM before producing any stage
result; the total-row denominators, A/B membership and mathematical estimands
are unchanged.

## 8. Frozen comparator predictors

For Full, Parallel and Orthogonal residual candidates, compute:

1. Raw common-gradient cosine.
2. Raw validation alignment using `-eta * c_t * R_candidate`.
3. A fixed-denominator Adam approximation: the candidate-induced first-moment
   difference divided by the bias-corrected denominator produced by the
   Blocked candidate.  This holds the second-moment denominator fixed and does
   not include candidate-specific square terms.
4. Exact AdamW finite-difference displacement.
5. Exact displacement norm as an unsigned magnitude diagnostic.
6. Sign-flipped residual as a directional negative control.

All formulas, parameter inclusion rules, epsilon placement, bias correction
and weight-decay cancellation checks must pass synthetic unit tests before a
real-model update is authorized.

## 9. Statistical unit and primary criteria

The training seed is the only confirmatory unit.  Three stages, three component
candidates and two probe directions are repeated measurements within seed.

The following rules are frozen before independent outcomes are read:

1. **Full local effect:** the seed-level mean `Y_R` is negative for at least
   four of five seeds and the five-seed mean is negative.
2. **Coordinate comparison:** within each seed, mean absolute prediction error
   of the exact AdamW predictor over the frozen repeated-measurement grid is
   lower than both raw validation alignment and the fixed-denominator Adam
   approximation; this must hold in at least four of five seeds.
3. **Directional control:** within each seed, form the frozen 18-observation
   grid from three stages, three residual candidates and two probe directions.
   Compute Kendall's tau-b between the observed effects and (a) the exact
   signed AdamW predictions and (b) the no-fit magnitude-only scores
   `q_mag = -||delta_candidate||_2`.  The negative sign freezes the simple
   alternative that a larger residual-attributable displacement is always
   more beneficial.  Require `tau_exact > tau_magnitude` in at least four of
   five seeds.  Ties use tau-b and observations whose absolute outcome is at
   most `1e-8` are retained as ties.  Separately, the seed-level mean outcome
   of Full must be more favorable than that of the sign-flipped residual in at
   least four of five seeds and in the five-seed mean.  Signed prediction
   accuracy and both tau values are reported for every seed, but no threshold
   is selected from their observed values.
4. **Stage robustness:** Full's mean outcome across the five seeds and two
   probe directions must be negative at at least two of the three frozen
   stages.  Separately, stage-pooled exact-predictor MAE must be lower than
   both comparator MAEs at at least two of the three stages.  These are
   anti-single-stage gates, not additional independent replicates.

No significance claim is inferred solely from a 4/5 directional rule.

Optimizer-induced non-additivity may be discussed only when the seed-level
mean `||iota|| / max(||delta_R||, 1e-30)` exceeds `1e-3` and the seed-level
mean `<v_probe, iota>` has the same sign as the Full-vs.-component-combination
outcome deviation in at least four of five seeds.  This is a conditional
secondary claim and is never required for the primary Exp64 result.

## 10. Failure and stopping rules

Exp64 fails if any of the following holds:

- Full is favorable in no more than three independent seeds;
- exact AdamW attribution does not beat simple raw validation alignment;
- the result appears only in already-observed EduBench states;
- the result depends on one stage;
- it appears only after changing seeds, stages, scale, probe partition or
  optimizer hyperparameters;
- it requires treating 15 states or individual records as independent units.

After the frozen independent run, no additional seeds, stages, scales,
datasets or method variants may be added.  A positive result supports a narrow
optimizer-state-aware local mechanism claim.  A negative result ends the
mechanism search and triggers the pre-declared CCF-C paper route.

## 11. Required preflight sequence

1. Close the Exp63 checkpoint-RNG audit under deterministic CUDA execution.
2. Materialize and hash the SummEval A/B group split.
3. Unit-test exact AdamW finite differences against direct optimizer copies.
4. Verify shared scaling and zero clipping on synthetic tensors.
5. Run one no-outcome real-model geometry/state preflight on an already
   observed Exp63 checkpoint.
6. Freeze source hashes and convert the protocol status from draft to formal.
7. Only then train the five independent DRB base trajectories.
