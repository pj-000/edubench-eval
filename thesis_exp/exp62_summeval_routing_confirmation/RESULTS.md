# Exp62 final result

## Decision

**Directional replication; confirmatory gates not passed.**

The twenty fixed-epoch checkpoints passed the mechanical integrity audit. The
test split was then accessed exactly once under the frozen protocol. No rerun,
alternative split, seed replacement, checkpoint selection, or threshold
change is authorized.

## Mean test results across five seeds

| Arm | Macro MAE | Coherence MAE | Fluency MAE |
|---|---:|---:|---:|
| Direct-Residual-Blocked | 0.529211 | 0.709003 | 0.349419 |
| Routed-HMSA | 0.520077 | 0.691537 | 0.348617 |
| Orthogonal-only | **0.514446** | **0.683416** | 0.345476 |
| Parallel-only | 0.532286 | 0.721447 | **0.343125** |

The primary metric is the unweighted macro average of coherence and fluency
expected-score MAE. Lower is better.

## Frozen paired comparisons against Direct-Residual-Blocked

| Comparison | Mean delta MAE | Favorable seeds | Source-cluster 95% interval | Gate |
|---|---:|---:|---:|---|
| Routed-HMSA minus DRB | -0.009134 | 4/5 | [-0.028093, 0.009428] | Fail |
| Orthogonal-only minus DRB | -0.014765 | 4/5 | [-0.031181, 0.000302] | Fail |
| Parallel-only minus DRB | +0.003075 | 2/5 | [-0.004533, 0.010191] | “No reproduced improvement” passes |

Negative deltas favor the candidate arm. The full-routing and orthogonal-only
intervals both cross zero, so neither comparison satisfies the preregistered
independent-confirmation rule. The orthogonal interval is close to, but still
not below, zero; it must not be rounded or described as a pass.

## Claims authorized by Exp62

- On an independent true multi-rater ordinal dataset, full residual routing
  and orthogonal-only routing show favorable mean directions relative to
  direct residual blocking in four of five seeds.
- Parallel-only does not reproduce that favorable pattern.
- The external pattern is suggestive and geometrically consistent with the
  EduBench observation, but source-article-cluster uncertainty prevents a
  confirmatory cross-dataset claim.
- The mean improvement is concentrated in coherence; the per-dimension
  contrast is descriptive, not a separately preregistered causal test.

## Claims not authorized

- Exp62 does not prove that residual routing generally improves ordinal
  prediction across datasets.
- Exp62 does not independently confirm that the orthogonal component is the
  cause of the improvement.
- The near-zero upper bound for Orthogonal-only is not a statistically or
  procedurally valid reason to change the bootstrap, split, seeds, or gate.

## Canonical artifacts

- `thesis_exp/outputs/exp62_summeval_routing_confirmation/stage0/stage0_audit.json`
- `thesis_exp/outputs/exp62_summeval_routing_confirmation/decision/formal_training_integrity.json`
- `thesis_exp/outputs/exp62_summeval_routing_confirmation/test_once/test_results.json`
- `thesis_exp/outputs/exp62_summeval_routing_confirmation/decision/test_access_record.json`
