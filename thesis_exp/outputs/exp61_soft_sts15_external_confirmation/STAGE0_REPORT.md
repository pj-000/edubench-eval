# Exp61 Soft-STS-15 Stage 0 report

## Decision

`CONDITIONAL_GO_TO_PROTOCOL_DESIGN`

This decision authorizes protocol and trainer design only. It does not authorize
GPU training or access to model outcomes on the frozen test split.

## What this dataset measures

Soft-STS-15 is a single-dimension English semantic textual similarity task. It
does not contain the twelve EduBench educational evaluation criteria. Its value
for Exp61 is structural: every record exposes five ordinal scores in `0..5`, so
the residual-alignment mechanism can be tested on an independent task with a
richer empirical rating distribution.

## Source closure

- Upstream repository: `ale0xb/sts_beyond_averages`
- Frozen commit: `ca754a21e58437c4b843d10161d2838f39230e7f`
- Frozen `data/text.clean` SHA-256:
  `8707cdb85e2da9818bc8c08cd90b9b2c17c483e08f7bb7ff5c6dc69b555dde71`
- Rows: 7,890
- Published score-list length: five for all 7,890 rows
- Invalid published scores: zero

The original SemEval preprocessing can collect more than five annotations and
then retain only the first five values in `score_list`. Consequently,
`n_annotators` ranges from 5 to 16, and the upstream `gs_score` differs from the
mean of the five published values for 368 rows. Exp61 therefore defines its
distribution, human mean, and rounded hard target from the same five published
values. It never combines `gs_score` with the five-score empirical distribution.

## Split audit

The upstream item-level 60/20/20 split is not accepted for Exp61 because it has:

- 7 canonical sentence pairs crossing splits;
- 774 normalized sentences crossing splits;
- 563 sentence-connected components crossing splits.

Exp61 instead freezes a deterministic sentence-connected-component-disjoint,
stratified split:

| Split | Rows | Components | Observed strata |
|---|---:|---:|---:|
| Train | 4,733 | 3,169 | 20 |
| Dev | 1,578 | 1,057 | 20 |
| Test | 1,579 | 997 | 20 |

There is zero sentence, canonical-pair, or sentence-component overlap across the
frozen splits. The largest absolute deviation in stratum proportion is below
0.0006 in every split.

## Remaining gates before training

1. Freeze the exact six-class trainer and three experimental arms.
2. Freeze hard-head inference and a scale-aware primary effect threshold.
3. Freeze component-cluster bootstrap and secondary non-inferiority metrics.
4. Complete dev-only no-update and real-model preflight checks.
5. Freeze model, tokenizer, environment, data, analysis, and source manifests.
6. Document upstream data use and avoid redistributing raw sentence text.

The machine-readable audit and split manifest are the authoritative artifacts.

