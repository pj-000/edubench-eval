# SORC-DPO one-time test results

## Execution status

- Execution date: 2026-07-30
- Reviewed execution commit: `7fae2f7`
- Result audit: `TEST_RESULT_PASS`
- Frozen test records: 2,218
- Completed runs: 12/12 (`P0/P1/P2/P3 × seeds 42/43/44`)
- Frozen Git test blob: `7749c2c0f166186cc840409d64424b5a78e7222a`
- Inference protocol: `RAR_SFT_VLLM_COMPACT_JSON_V1`
- Bootstrap: 10,000 paired record-cluster replicates, seed `20260731`
- Primary multiplicity: six tests with Holm–Bonferroni correction
- Strict parse rate: 1.0 for every arm and seed
- Test rerun allowed: no

## Aggregate test results

| Arm | MAE↓ | L2H↓ | Exact↑ | Kendall↑ | Bias | Label-2 recall↑ | Label-5 recall↑ | QWK↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 R3-SFT | 0.3569 | 63.11% | 0.7218 | 0.5854 | +0.1673 | 0.00% | 85.18% | 0.5336 |
| P1 Field-DPO | 0.3324 | 46.93% | 0.7286 | 0.6117 | +0.1274 | 1.42% | 84.82% | 0.6290 |
| P2 SORC-score | 0.3293 | 44.98% | 0.7299 | 0.6135 | +0.1204 | 1.42% | 84.65% | 0.6394 |
| P3 joint SORC | 0.3230 | 41.75% | 0.7311 | 0.6202 | +0.1189 | 1.42% | 84.79% | 0.6586 |

Each table entry is first computed within seed and then averaged equally
across seeds 42, 43, and 44.

## Preregistered component contrasts

Positive values mean benefit under the endpoint-specific sign convention.

| Contrast | MAE benefit [95% CI] | Holm p | L2H benefit [95% CI] | Holm p | Classification |
|---|---:|---:|---:|---:|---|
| P1 − P0 | +0.0245 [+0.0170, +0.0326] | 0.0012 | +0.1618 [+0.1255, +0.2013] | 0.0012 | Strong support |
| P2 − P1 | +0.0032 [−0.0003, +0.0068] | 0.0806 | +0.0194 [+0.0031, +0.0382] | 0.0740 | Directional support |
| P3 − P2 | +0.0063 [+0.0018, +0.0111] | 0.0168 | +0.0324 [+0.0125, +0.0543] | 0.0128 | Strong support |

All six primary endpoints had 10,000 valid bootstrap replicates. No
preregistered operational failure, material guardrail harm, or forced-close
diagnostic harm was observed.

## Scientific interpretation

1. **The main preference result is P1.** Relative to the frozen R3-SFT cold
   start, preference pairs constructed from the model's actual train-only
   scoring errors produced statistically supported improvements in both
   overall MAE and catastrophic low-score overestimation. This supports the
   thesis claim that actual-error-driven, field-local preference optimization
   adds value beyond rationale-aware SFT.

2. **The ordinal offset has only directional evidence.** P2 is numerically
   better than P1, but neither co-primary survives the six-test Holm family at
   0.05. The result must not be described as a confirmed independent SORC
   offset effect.

3. **P3 has a supported score-side bundled increment, not a reasoning
   mechanism result.** P3 is not FLOP-matched to P2. Its contrast adds both the
   rationale preference block and extra token/FLOP exposure. The test therefore
   supports only that bundled score-side increment.

4. **Visible rationale quality did not receive supporting evidence.** The
   earlier two-agent blind rationale audit was approximately zero. The present
   score results cannot override it, so the thesis must not claim that P3
   improves rationale quality, rationale semantic alignment, or internal
   reasoning.

5. **The low-score tail remains unresolved.** P3 lowers L2H from 63.11% to
   41.75%, but mean Label-2 recall remains 1.42%. The method mainly prevents
   severe promotion of low-quality answers into scores 4/5; it does not yet
   identify score 2 reliably.

## Public result bindings

| Artifact | SHA-256 |
|---|---|
| `final_results.json` | `ba738748f77b95f4be73bafcd4b9a775dfb5cf48e7ebb1988335eeee316bb735` |
| `multiseed_summary.csv` | `b5d01490e9a38ce5fdbeb5a37e736f9cc683d33c7ec987db18aa7beb8484eb85` |
| `paired_bootstrap.json` | `4738497d815b600d4c13b12e4d421cbe178369f22749addf434015b1f30bf50b` |
| `per_seed_metrics.csv` | `a676c71a6a8db6c484be51ecdbeb01ac0180b50b1c73c88b3eb5ecc9a3700a31` |
| `report.md` | `cfe0745ea45c9194e05ebacf127f18092a529a80a6a1689ce8ec7fb15a6adfda` |

Only aggregate metrics and statistical summaries are public. The test source,
row-level predictions, visible rationales, and prompt/output payloads remain
private and are not committed.
