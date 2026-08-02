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
   - `thesis_exp/configs/exp57_cbrd/stage1_source_lock.json`
3. Aggregate decisions:
   - `decision/stage1_development_decision.json`
   - `decision/stage1_pilot_decision.json`
4. Integrity and implementation audits:
   - `audit/stage1_final_integrity_audit.json`
   - `audit/posthoc_checkpoint_identity_training_kernels.json`
   - `audit/posthoc_checkpoint_identity_deterministic.json`
5. Per-run evidence:
   - each run's `selected_dev_metrics.json`, `dev_metrics_history.json`,
     `run_summary.json`, and `training_trace_first64.json`

Model checkpoints and repeated 664-row prediction dumps are intentionally not
stored in this Git review bundle.  The final integrity audit records the
prediction-count, record-ID, metric-recomputation, checkpoint-selection, and
test-access checks performed before publication.

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

## Evidence boundary for review

The current evidence supports the narrower statement that the aligned
human-disagreement residual carries useful development-set supervision beyond
a duplicated hard-label auxiliary task and a fixed shuffled residual.  It does
not yet establish a CCF-B-level general method, independent-dataset
generalization, fully passed directional causality, or final test performance.

The next decision should be whether this mechanism result is strong and novel
enough to justify additional seeds and a final test protocol, or whether it is
better positioned as a mechanistic explanation and ablation section inside
the existing HMSA paper.
