# Exp57 CBRD review guide

## Purpose

Exp57 tests a mechanism hypothesis derived from the completed HMSA result.  It
does not claim that the cross-entropy decomposition is itself novel.  For a
consensus label `y`, the empirical distribution of three human ratings `d`,
and auxiliary logits `z`, the exact identity is

`CE(d, z) = CE(e_y, z) + (e_y - d)^T z`.

The experiment asks whether the signed residual term provides useful
boundary-directed supervision beyond (i) a generic second hard-label head and
(ii) an unaligned residual control.

All reported Exp57 results are development-set results.  The historical test
set was not accessed (`test_access_count = 0`).

## Evidence to inspect

1. Method and implementation:
   - `thesis_exp/exp57_cbrd/README.md`
   - `thesis_exp/exp57_cbrd/STAGE1_PROTOCOL.md`
   - `thesis_exp/exp57_cbrd/method.py`
   - `thesis_exp/exp57_cbrd/losses.py`
   - `thesis_exp/exp57_cbrd/train.py`
   - `thesis_exp/exp57_cbrd/gate.py`
2. Frozen protocol and source locks:
   - `thesis_exp/configs/exp57_cbrd/stage1_protocol.json`
   - `thesis_exp/configs/exp57_cbrd/stage1_confirmation_protocol.json`
   - `thesis_exp/configs/exp57_cbrd/stage1_source_lock.json`
3. Aggregate decisions:
   - `decision/stage1_development_decision.json`
   - `decision/stage1_confirmation_decision.json`
   - `decision/stage1_pilot_decision.json`
4. Integrity and implementation audits:
   - `audit/stage1_final_integrity_audit.json`
   - `audit/stage1_primary_common_epoch.json`
   - `audit/stage1_confirmation_common_epoch.json`
   - `audit/stage1_confirmation_question_bootstrap.json`
   - `audit/stage1_clip_gradient_audit.json`
   - `audit/stage1_confirmation_checkpoint_hashes.json`
   - `audit/posthoc_checkpoint_identity_training_kernels.json`
   - `audit/posthoc_checkpoint_identity_deterministic.json`
5. Per-run evidence:
   - each run's `selected_dev_metrics.json`, `dev_metrics_history.json`,
     `run_summary.json`, and `training_trace_first64.json`

Model checkpoints and repeated prediction copies are intentionally not stored
in this Git review bundle.  One canonical 664-row selected-dev prediction file
is included for every run in the five-seed primary pair, together with complete
epoch histories and SHA-256 hashes for all ten selected checkpoints.  This is
enough to recompute the primary metrics and question-cluster bootstrap without
publishing roughly 12 GB of model weights.

## Three-seed development results

| Variant | MAE (lower) | Exact (higher) | Kendall's tau (higher) |
|---|---:|---:|---:|
| Consensus-only | 0.396084 | 0.722892 | 0.572330 |
| Dual-hard | 0.389056 | 0.722390 | 0.584943 |
| Routed-HMSA | **0.373829** | 0.728916 | **0.611301** |
| Residual-only | **0.370649** | **0.731426** | 0.607348 |
| Shuffled residual | 0.394578 | 0.719880 | 0.573549 |
| Sign-flipped residual | 0.405120 | 0.712851 | 0.562690 |

Primary comparison, Routed-HMSA minus Consensus-only:

- mean delta MAE: `-0.022256`; all three seeds favor Routed-HMSA;
- mean delta Exact: `+0.006024`;
- mean delta Kendall's tau: `+0.038972`.

Additional controls:

- Routed-HMSA minus Dual-hard: mean delta MAE `-0.015228`;
- Routed-HMSA minus Shuffled-residual: mean delta MAE `-0.020750`, favorable
  in all three seeds;
- Residual-only minus the historical Hard-only reference: mean delta MAE
  `-0.022423`, favorable in all three seeds.

## Gate outcome and mandatory qualifications

Four of five development gates passed:

- primary residual increment: pass;
- beyond generic dual-hard regularization: pass;
- aligned residual beats fixed shuffle: pass;
- residual-only sufficiency: pass;
- sign direction: **fail under the preregistered all-strata rule**.

The sign-flipped comparison is favorable in pooled analysis, including the
question-cluster bootstrap (`95% CI [2.6184, 4.4890]` for the registered
margin statistic), but the upward-boundary stratum at seed 44 is slightly
negative (`-0.0225`).  It must therefore not be reported as a fully passed
directionality claim.

The historical implementation-parity pilot is also `PILOT_NO_GO`:

- Routed-HMSA vs. ordinary HMSA: absolute delta MAE `0.003514`, prediction
  agreement `89.76%`;
- detached-soft vs. Hard-only: absolute delta MAE `0.003514`, prediction
  agreement `89.31%`.

Post-hoc gradient audit found exact equality under deterministic kernels, but
a maximum shared-backbone gradient difference of `1.2207e-4` under the
default training kernels.  The detached control is additionally affected by
global gradient clipping because the otherwise detached auxiliary head still
contributes to the global norm.  This is a qualification of the parity gate
and detached control; it must not be hidden or silently reclassified as a
pass.

## Five-seed primary-pair confirmation

The extension to seeds 45 and 46 was frozen before either result was read and
authorized only four new runs: Consensus-only and Routed-HMSA at each seed.
All five seeds favor Routed-HMSA on MAE.

| Seed | Delta MAE | Delta Exact | Delta Kendall's tau |
|---:|---:|---:|---:|
| 42 | -0.022088 | +0.003012 | +0.042934 |
| 43 | -0.033133 | +0.006024 | +0.046855 |
| 44 | -0.011546 | +0.009036 | +0.027126 |
| 45 | -0.046185 | +0.021084 | +0.052737 |
| 46 | -0.009036 | +0.001506 | +0.002957 |
| Mean | **-0.024398** | **+0.008133** | **+0.034522** |

All frozen five-seed gates passed.  Forty-two of 50 same-epoch seed/epoch
cells favor Routed-HMSA on MAE; the mean epoch-10 delta is `-0.021888`, so the
effect is not confined to independently selected checkpoints.  The paired
question-cluster bootstrap over the five-seed mean prediction effect gives
delta MAE `-0.024398` with a 95% interval of `[-0.037004, -0.012546]`.

This decision is recorded as `CONFIRMATION_PASS_INTERNAL_MECHANISM`.  It is an
internal development-set confirmation, not external generalization and not a
new sealed-test result.

## Global-clipping interaction audit

The strict audit replays a complete 32-microbatch accumulation window at three
initial states and six selected checkpoints.  It performs no optimizer step.
The cuBLAS-deterministic replay exactly matches the first diagnostic replay.

At initialization, the aligned/consensus clip-coefficient ratio ranges from
`1.0011` to `1.0127`.  At selected checkpoints it ranges from `1.0895` to
`1.4910`.  In all six selected states, the residual is anti-aligned with the
consensus-route total gradient (cosine range `[-0.7689, -0.3975]`), reducing
the total pre-clip norm and thereby increasing the global clip coefficient.

Consequently, the experiment identifies the effect of the residual-routing
intervention together with its interaction with the frozen optimizer and
global clipping.  It does **not** isolate a scale-matched residual direction
in pure form.  A pure direction-only causal claim requires a preregistered
matched-clipping control; alternatively, the paper must retain the narrower
joint-intervention claim.

## Evidence boundary for review

The current evidence supports the narrower statement that the aligned
human-disagreement residual-routing intervention, under the frozen optimizer
and global clipping, consistently improves development-set performance beyond
a duplicated hard-label auxiliary task, consensus-only routing, and one fixed
shuffled residual.  It does not yet establish a CCF-B-level general method,
independent-dataset generalization, scale-matched direction-only causality,
fully passed signed directionality, or final test performance.

The primary effect no longer needs additional EduBench seeds.  The next
decision is whether to (i) keep the result as a strong HMSA mechanism section,
or (ii) pursue a preregistered matched-clipping control followed by a minimal
primary-pair replication on an independent multi-rater ordinal dataset.  The
historically used EduBench test must not be presented as the sole independent
confirmation.
