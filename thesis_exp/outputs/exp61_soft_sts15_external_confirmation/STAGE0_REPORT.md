# Exp61 Soft-STS-15 revised Stage 0 report

## Decision

`EXP61_STAGE0_REVISED_GO_TO_PROTOCOL_FREEZE`

This authorizes the next protocol-freeze and implementation-preflight stage. It
does **not** authorize GPU training or test-model evaluation.

## Scientific role

Soft-STS-15 is a single-dimension English semantic textual similarity task. It
does not contain the twelve EduBench educational criteria. Exp61 uses it only to
test whether sample-aligned non-collinear residuals outperform a
geometry-matched misaligned residual on an independent multi-rater ordinal task.

## Source closure

- Upstream repository commit:
  `ca754a21e58437c4b843d10161d2838f39230e7f`
- Public 7,890-row data SHA-256:
  `8707cdb85e2da9818bc8c08cd90b9b2c17c483e08f7bb7ff5c6dc69b555dde71`
- Official SemEval archive SHA-256:
  `c1c175b30e570a3995a79ca0f6169223682ba1f1e0788822554df9c79c7fdaf1`
- All four relevant official Perl scripts are individually hash-locked.

The verified official `find_pairs.pl` computes `gs_score` from all collected
scores and then retains the first five encountered scores in the published
score list. The public file therefore exposes five integer scores for every row,
while `n_annotators` ranges from 5 to 16. For 368 rows, `gs_score` differs from
the mean of the five published scores.

Exp61 excludes `gs_score`. Its soft target, human mean, quantized hard target,
and evaluation reference all come from the same five published scores. The five
scores are not described as the complete set of annotations originally
collected or as an annotator-population distribution.

The code license is MIT, but that does not independently license redistribution
of the SemEval sentence text. Exp61 publishes code and row/hash split metadata,
not a repackaged raw-text dataset.

## Hard-target semantics

The hard target is

\[
y_i=\left\lfloor \frac{1}{5}\sum_{j=1}^{5}r_{ij}+0.5\right\rfloor.
\]

It is called the **quantized-mean main target**, not an unqualified consensus or
majority label. The full audit finds:

- 1,188 rows without a unique mode;
- 1,373 disagreements between the quantized mean and unique mode among the
  6,702 rows that do have a unique mode;
- 548 rows where the quantized target is absent from the five published scores;
- 1,246 disagreements between the quantized target and median;
- 5,910 non-zero residual targets;
- 209 distinct empirical rating distributions.

The residual is consequently named the *residual relative to the
quantized-mean main target*. It can span several ordinal levels and is not
described as a minority-vote boundary residual.

## Split revision

The upstream item-level split is not used for confirmation because it contains
7 cross-split canonical pairs, 774 cross-split sentences, and 563 cross-split
sentence components.

The first Stage 0 component split was also rejected because one 348-row
component occupied 22.0% of test. The revised procedure enumerates all 20
two-stage StratifiedGroupKFold combinations without model predictions, filters
them using frozen integrity and composition constraints, and selects by a fixed
lexicographic rule.

Selected candidate: outer fold 4, inner fold 2.

| Split | Rows | Components | Largest component | Share |
|---|---:|---:|---:|---:|
| Train | 4,734 | 3,109 | 348 | 7.35% |
| Dev | 1,578 | 1,057 | 47 | 2.98% |
| Test | 1,578 | 1,057 | 47 | 2.98% |

All 20 observed mode/agreement strata occur in every split. Cross-split
sentence, canonical-pair, and connected-component overlap are all zero. The
split-manifest SHA-256 is:

`73b04b02c3aaac303e8ba3b9213030a9dcdf1e5736f18bf3f81a987be8da56ad`

The manifest includes a hash of each five-score target plus its derived human
mean and hard label, but it does not redistribute raw sentence text or scores.

## Fail-closed gates

The audit now stops before producing a pass artifact if any required invariant
fails, including upstream commit/data/archive/script hashes, row count, score
count, score domain, row-ID uniqueness, split completeness, all-strata coverage,
sentence/pair/component isolation, evaluation component share, deterministic
split replay, or manifest replay and completeness.

## Tokenizer audit

The input template contains only Sentence 1, Sentence 2, and a fixed 0–5
semantic-similarity instruction. It excludes origin, mode, agreement, ratings,
and split metadata.

The exact Exp60 Qwen3-Reranker-0.6B tokenizer snapshot is hash-locked. Across all
7,890 rendered inputs:

- median: 59 tokens;
- p95: 80 tokens;
- p99: 96 tokens;
- maximum: 143 tokens.

The smallest frozen candidate that avoids all truncation is `max_length=256`.
No model weights were loaded, no training occurred, and no GPU was used.

## What remains before training

1. Freeze the exact three-arm protocol and scale-aware confirmation gates.
2. Implement and freeze the six-class trainer and maximum-mismatch mapping.
3. Freeze inference, component-bootstrap analysis, and single test-unsealing code.
4. Complete CPU static checks and real-model no-update geometry preflights.
5. Freeze source, environment, model, tokenizer, data, and analysis manifests.

Only a subsequent preflight decision may authorize the nine formal GPU runs.
