# Exp54 frozen-prediction statistical closure

Status: analysis specification fixed before running the new bootstrap. This is
an explicitly post-review statistical correction, not a new preregistration of
the original experiments. Historical results remain unchanged.

## Scope and authority

- Source revision: `a75f79d` (RAR-SFT / preference / mechanism evidence).
- Only existing epoch-3 S0/R1/R2/R3 dev predictions, final learning-rate-5e-6
  P1/P2/P3 dev predictions, the four-arm frozen test, and the three post-hoc
  mechanism-control test predictions are used. Seeds are exactly 42/43/44.
- No training, model loading, generation, checkpoint selection, GPU, evaluator,
  new probability extraction, or hyperparameter changes. Test metadata and
  already-generated predictions are reread for this analysis; do not report
  `test_accessed=false`.
- HMSA and its follow-ups are outside this bounded correction.

## Estimand and resampling

The point estimand is the mean of three independently computed seed-specific
metrics on all records. It is **record-weighted**, not equal-weighted per QA or
question, and not a metric of ensembled predictions. Each contrast is treatment
minus baseline; negative MAE/L2H means improvement. Report every paired seed
difference and sample SD (ddof=1). Confidence intervals are conditional on these
three trained models, not unconditional over all possible training seeds.

1. Primary resampling unit: the split's existing `question_key`.
2. Sensitivity: canonical `(question_key, answer_key)` QA tuple.
3. Diagnostic only: record-level resampling with the *same fixed-seed estimand*,
   to separate cluster effects from the historical SFT seed-resampling change.

Sample K clusters uniformly with replacement from the K observed clusters;
retain every record and every repeated copy of each sampled cluster. The same
draw is used across every arm and seed within a split. Recompute each metric on
the full resampled confusion matrix for each seed, then average the three
metrics. Never average per-cluster Kendall or QWK. Unequal cluster sizes yield
variable bootstrap row counts, and the metric denominator follows those counts.

Use 10,000 replicates and NumPy PCG64 seed 20260905, processed in fixed blocks of
100. Report two-sided percentile 95% CIs, valid replicate counts and undefined
counts. A missing subgroup or degenerate rank/kappa denominator is undefined,
not zero. With fewer than 9,500 jointly finite replicates, omit inferential
claims for that endpoint. Report the number of clusters supporting each class,
because row counts alone overstate low-score support.

For descriptive approximate two-sided tests, use the null-centered bootstrap:
`p = (1 + count(abs(delta_boot - delta_observed) >= abs(delta_observed))) /
(1 + valid_replicates)`. This is an asymptotic bootstrap approximation, **not an
exact randomized experiment test**. It differs from the historical sign-tail
approximation; both the method and CI change must be disclosed. Bootstrap
intervals are unadjusted; Holm-adjusted p-values are a separate summary, not
simultaneous confidence intervals.

## Comparisons and multiplicity (fixed without inspecting new intervals)

Within each family, primary endpoints are MAE and L2H_rate. The test families
retain the original collectors' three-comparison/two-endpoint structure.

| Family | Baseline → treatment | Status |
|---|---|---|
| SFT_DEV | S0 → R3; R1 → R3; R2 → R3 | 6 tests, exploratory; dev used for epoch selection |
| PREFERENCE_DEV | R3 → P1; P1 → P2; P2 → P3; R3 → P3 | 8 tests, developmental/exploratory |
| FORMAL_TEST | R3 → P1; P1 → P2; P2 → P3 | 6 tests, correction of original frozen-test comparisons |
| MECHANISM_TEST | TOKENAVG → R3; FULLSEQ → P1; SYN → P1 | 6 tests, post-hoc on previously accessed test |

Holm is applied to each full family (undefined endpoints conservatively enter
adjustment as p=1, and remain unclaimable). QA/record analyses receive their own
adjustments only for sensitivity reporting. Additionally report a pooled
12-test Holm sensitivity over FORMAL_TEST + MECHANISM_TEST; this does not turn
post-hoc controls into confirmatory evidence. No endpoint, contrast, seed, arm,
or cluster definition will be selected according to significance.

## Metrics, validation, and disclosure

Report Exact, MAE against discrete label_5, signed and absolute signed bias,
Kendall tau-b, quadratic weighted kappa, L2H/H2L counts and rates, all five label
recalls with integer per-seed numerators/denominators, 5×5 confusion matrices,
strict parse rate, and forced-close rate. Report arm means/sample SD and paired
contrast intervals for all rates/score metrics; non-primary endpoints are
exploratory. Do not derive NLL/Brier/RPS from argmax scores.

Validate exact row identity/order/label against the locked split; no silently
dropped failed parses or forced-close outputs. Match recomputed point metrics
to historical public reports. Verify available completion receipt prediction
and protocol hashes and existing public source anchors. Record any provenance
level not backed by a pre-existing individual prediction hash explicitly.

Publish aggregate results and whole-file hashes only, never row IDs, cluster
keys, generated rationales, or per-row predictions. Original source collectors,
locks and reports stay unchanged. RAR comparisons remain dev evidence; P3 is
a bundled training intervention, not an isolated rationale-preference effect;
TOKENAVG/FULLSEQ/SYN remain post-hoc. Statistical closure cannot remove these
design limits or establish automatic graduation eligibility.
