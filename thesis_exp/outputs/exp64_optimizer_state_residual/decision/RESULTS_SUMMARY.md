# Exp64 optimizer-state-aware residual attribution: final result

## Frozen decision

**`EXP64_NO_GO_STOP`**

Exp64 completed the single preregistered five-seed confirmation run and must
stop under the frozen protocol. The decision is a scientific NO-GO for the
strong claim that the exact AdamW-attributable residual displacement is a
consistently better local explanatory coordinate than raw validation-gradient
alignment. It is not a mechanical failure and does not erase the narrower
directional findings below.

## Completion and integrity

- Independent seeds: 72--76 (5/5 complete).
- Frozen stages per seed: epochs 2, 5 and 8 (15/15 complete).
- Cross-probes: source-group-disjoint A-to-B and B-to-A.
- Formal optimizer step calls: 0; all arms are restored-state one-step
  counterfactuals.
- Test access count: 0.
- Decision file SHA-256:
  `b4b748def3ebcaa93f8ffe9eb1e1362456b9adf97ffbbb5c07e4047abd199583`.

## Gate results

| Frozen gate | Result | Evidence |
|---|---:|---|
| Full residual has a favorable local effect | PASS | 4/5 seeds; overall mean hard-loss change = -0.001394 |
| Exact AdamW predictor beats raw validation alignment | **FAIL** | 3/5 seeds; required at least 4/5 |
| Exact AdamW predictor beats fixed-denominator Adam approximation | PASS | 5/5 seeds |
| Exact signed direction beats displacement magnitude | PASS | 4/5 seeds |
| Full residual beats the sign-flipped control | PASS | 5/5 seeds; Full mean = -0.001394, sign-flipped mean = +0.003914 |
| Full effect is not confined to one stage | PASS | favorable at epochs 2 and 5, unfavorable at epoch 8 |
| Coordinate result is not confined to one stage | PASS | exact beats both comparators at epochs 2 and 5 |
| Optimizer nonadditivity claim authorized | **NO** | prediction/outcome sign agreement in 2/5 seeds; required 4/5 |

Negative hard-loss change is favorable because every arm is compared with the
same blocked-residual counterfactual at the same frozen state.

## What the experiment supports

1. The signed full residual has a reproducible local directional effect under
   the frozen intervention: it is favorable in four of five independent seeds
   and is more favorable than reversing its sign in all five seeds.
2. Exact AdamW state matters relative to the deliberately crude
   fixed-denominator approximation and contains more directional information
   than displacement magnitude alone.
3. The effect is stage-dependent: it is favorable at epochs 2 and 5 but not at
   epoch 8. It therefore cannot be described as universally beneficial.

## What the experiment does not support

1. Exact AdamW finite-difference attribution is not a consistently superior
   explanation to the simpler raw validation-alignment predictor. The frozen
   primary threshold was missed by one seed (3/5 instead of 4/5).
2. AdamW-induced nonadditivity between the parallel and orthogonal residual
   components is not authorized as a beneficial mechanism claim.
3. A one-step local result does not explain the full long-run HMSA test gain and
   does not establish a general optimizer-aware method.

## Consequence for the paper

Exp64 closes the preregistered mechanism search. It should be reported as a
boundary result rather than used to create a new positive CCF-B claim. The
paper can retain the strongest controlled evidence—signed full-residual
directionality and the sign-flip control—but must explicitly state that exact
AdamW coordinates did not consistently outperform raw validation alignment.
Under the frozen plan, no additional seeds, stages, scales, datasets or method
variants may be added to rescue the Exp64 hypothesis; the manuscript should
follow the declared CCF-C route unless a genuinely new study is designed as a
separate project.
