# SORC-DPO learning-rate follow-up results

## Question

The original preference run used a learning rate of `5e-7` for 27 optimizer
steps and produced a credible near-zero dev effect.  The train-only diagnostic
showed that the adapters had changed by only about `0.04%` in relative L2
norm.  This follow-up changed only the learning rate to `5e-6`; the frozen
pairs, loss, beta, offsets, optimizer-step count, scheduler, seeds, and
inference protocol were unchanged.

## Train-signal result

All nine P1/P2/P3 × seed 42/43/44 runs completed and passed the aggregate
training audit.  Relative adapter updates increased to `0.386%–0.442%`.
P1/P2 score preference contrast was about `0.11`; P3 score contrast was about
`0.092–0.103`, while P3 rationale contrast was `0.0229–0.0255` and positive
for `89.9%–90.5%` of rationale pairs.

This closes the original under-training diagnosis: the new adapters learned
the frozen preference pairs.  P2's strict ODPO offset-satisfaction rate
remained zero, so the learned score margin still did not reach the full
risk-dependent target.

### Train-only mechanism decomposition

An exact within-pair comparison of P2 against P1 refined the interpretation:

| Score block | Seed 42 | Seed 43 | Seed 44 | Mean P2−P1 β-margin |
|---|---:|---:|---:|---:|
| Adjacent score | -0.000292 | +0.000401 | -0.000583 | -0.000158 |
| Severe L2H | +0.005674 | +0.005674 | +0.005345 | +0.005565 |
| H2L guard | -0.005181 | -0.003536 | -0.003618 | -0.004112 |

The ordinal offset was therefore not inert: it consistently redirected train
margin toward severe low-to-high pairs.  However, no P2 score pair reached the
full prescribed offset, the high-score guard margin weakened relative to P1,
and the dev comparison did not establish an independent P2-over-P1 benefit.
This supports the presence of risk-conditioned optimization pressure, not the
claim that ODPO independently improved generalization.

## Frozen dev result

All nine dev runs used the same deterministic vLLM/XGrammar protocol as the
original preference experiment.  Each run evaluated the same 664 dev rows.
Test remained sealed.

| Arm | Exact ↑ | MAE ↓ | Signed bias | Kendall τ ↑ | L2H ↓ | Recall-1 ↑ | Recall-2 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 Field-DPO | 0.7103 | 0.3353 | +0.1034 | 0.5978 | 45.0% | 55.6% | 4.8% |
| P2 SORC-score | 0.7083 | 0.3373 | +0.0974 | 0.5963 | 43.3% | 61.1% | 4.8% |
| P3 Joint-SORC | 0.7088 | 0.3348 | +0.0969 | 0.6011 | 41.7% | 72.2% | 7.1% |

### P3 versus its R3-SFT cold start

| Metric | R3-SFT | P3 LR=`5e-6` | Difference |
|---|---:|---:|---:|
| Exact ↑ | 0.7038 | 0.7088 | +0.0050 |
| MAE ↓ | 0.3409 | 0.3348 | -0.0060 |
| Signed bias | +0.1280 | +0.0969 | -0.0311 |
| Kendall τ ↑ | 0.5911 | 0.6011 | +0.0101 |
| L2H ↓ | 51.7% | 41.7% | -10.0 pp |
| Recall-1 ↑ | 44.4% | 72.2% | +27.8 pp |
| Recall-2 ↑ | 2.4% | 7.1% | +4.8 pp |

The P3 L2H change was `-10.0 pp` for every seed.  Exact, MAE, bias, and
Kendall also moved in the favorable direction for all three seeds, although
the changes were small.

## Interpretation

The correct conclusion is:

> Increasing the preference-training learning rate closed the verified
> under-training problem and produced consistent directional improvement in
> low-score risk on dev.  P3 reduced low-to-high errors most, while preserving
> overall score quality.  Current paired bootstrap intervals still touch or
> cross zero, so the result is promising directional evidence rather than a
> statistically decisive superiority claim.

P3 versus R3 paired-bootstrap intervals were:

- Exact: `+0.0050`, 95% CI `[-0.0050, +0.0151]`;
- MAE: `-0.0060`, 95% CI `[-0.0211, +0.0105]`;
- L2H: `-0.0999`, 95% CI `[-0.2334, 0.0000]`;
- Recall-2: `+0.0482`, 95% CI `[0.0000, +0.1905]`.

The next scientific step is the already planned rationale blind audit.  Test
must remain sealed until the SFT and preference-method selections are frozen.

## P3-versus-P2 rationale blind audit

The fixed 40-record × three-seed audit found no robust P3 rationale-quality
advantage over P2. Tie-adjusted overall preferences were:

- score blind, Codex Sol: `0.517`, 95% CI `[0.454, 0.579]`;
- score blind, Codex Terra: `0.512`, 95% CI `[0.450, 0.575]`;
- score visible, Codex Sol: `0.504`, 95% CI `[0.450, 0.558]`;
- score visible, Codex Terra: `0.533`, 95% CI `[0.463, 0.600]`.

Thus P3 learned its frozen rationale preferences on train, but did not
establish a stable cross-agent improvement in visible dev rationale quality.
The rationale block must be reported as an informative near-zero ablation,
not as a confirmed reasoning or rationale-alignment improvement.
