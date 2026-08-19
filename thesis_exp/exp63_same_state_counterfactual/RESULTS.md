# Exp63 final result

## Decision

**Full-residual-only support; the preregistered isolated orthogonal-direction
claim does not pass.**

All five seeds, three fixed training stages per seed, and four counterfactual
arms per stage completed.  The confirmatory unit is a seed: the three fixed
stage contrasts were averaged within each seed before applying the frozen
five-seed decision rule.  The historical test split was never accessed.

## What Exp63 controls

For seeds 67--71, complete Consensus-only model, AdamW, scheduler, and RNG
states were frozen after epochs 2, 5, and 8.  From each exact state, all four
arms used the same next 128 training examples and differed only in the
backbone-gradient component:

| Arm | Counterfactual gradient before norm matching |
|---|---|
| Blocked | common gradient only, $G_C$ |
| Full residual | $G_C+R$ |
| Parallel-only | $G_C+R_{\parallel}$ |
| Orthogonal-only | $G_C+R_{\perp}$ |

Every full-parameter candidate was matched to a target norm of 0.95 and then
passed through the same global clipping threshold of 1.0.  The realized BF16
norm was 0.94921875 for every arm, so the effective clipping coefficient was
1.0 in every update.  Each arm then took exactly one AdamW step and was
evaluated on the complete frozen dev probe using hard-head cross-entropy.

## Frozen seed-level results

Lower hard cross-entropy is better.  Values below are candidate minus
reference after averaging the three fixed stages within each seed.

| Seed | Full residual $-$ Blocked | Orthogonal $-$ Blocked | Orthogonal $-$ Parallel |
|---:|---:|---:|---:|
| 67 | -0.000898 | +0.000107 | +0.000516 |
| 68 | -0.002973 | -0.000592 | +0.003128 |
| 69 | -0.004676 | -0.000580 | -0.000313 |
| 70 | +0.001468 | +0.000923 | -0.000666 |
| 71 | -0.000107 | -0.003245 | -0.003683 |
| **Mean** | **-0.001437** | **-0.000678** | **-0.000204** |
| **Favorable seeds** | **4/5** | **3/5** | **3/5** |

The frozen primary rule for Orthogonal-only versus Blocked required a mean
delta at most -0.00001 and at least four favorable seeds.  Its mean direction
was favorable, but only three seeds were favorable, so the primary comparison
fails.  Orthogonal-only versus Parallel-only also fails the same descriptive
thresholds (3/5 favorable seeds).

Full residual versus Blocked reaches the frozen directional thresholds:
mean delta -0.001437 with four of five favorable seeds.  This is a small local
one-step effect and is not a new external-generalization result.

## Authorized interpretation

- Matching the full-parameter norm and preventing clipping does not eliminate
  the mean advantage of the complete residual update.  Therefore gradient
  magnitude and unequal clipping alone are insufficient explanations for the
  observed local effect.
- The isolated orthogonal component is not stable enough across seeds and
  stages to support the stronger claim that it alone causes the benefit.
- The most defensible mechanism statement is that the complete signed
  residual can affect the optimizer update beyond duplicated consensus under
  controlled states, while its useful effect is not cleanly attributable to
  the orthogonal component alone.  Interaction between parallel and
  orthogonal components, optimizer state, and longer training trajectories
  remains plausible.
- Exp63 narrows the Exp59 interpretation; it does not invalidate the earlier
  long-run performance observation.

## Integrity audit

- Seed units: 5
- Fixed seed-stage units: 15
- Valid counterfactual updates: 60
- Positive checkpoint learning rates: 1.87630668e-5, 1.07845910e-5,
  and 2.09844988e-6
- Maximum target-norm absolute error: 0.00078125
- Maximum post-clipping norm: 0.94921875
- Maximum residual reconstruction relative error: 0
- Maximum normalized orthogonality error: 1.8241e-15
- Test access count: 0

An initial counterfactual execution constructed the scheduler after restoring
AdamW, which reset the learning rate to zero and produced no parameter update.
Those mechanically invalid no-op records were detected from exact equality to
the pre-update probe, stopped, and preserved under `failed_v1_lr_zero`.  They
are excluded from every result above.  The corrected implementation constructs
the scheduler first, then restores optimizer and scheduler state, and enforces
equality to a positive checkpoint learning rate before every arm.

## Canonical artifacts

- `thesis_exp/configs/exp63_same_state_counterfactual/protocol.json`
- `thesis_exp/configs/exp63_same_state_counterfactual/source_lock.json`
- `thesis_exp/outputs/exp63_same_state_counterfactual/preflight/real_model.json`
- `thesis_exp/outputs/exp63_same_state_counterfactual/decision/canonical_results.json`
- `thesis_exp/outputs/exp63_same_state_counterfactual/counterfactual/seed_*/after_epoch_*.json`
