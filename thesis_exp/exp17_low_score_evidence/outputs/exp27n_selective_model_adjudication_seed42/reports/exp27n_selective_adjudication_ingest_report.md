# Exp27N Selective Model Adjudication Ingest

The single GPT-5.6Pro session returned all 54 target-aware blind reviews.
These annotations are model-reviewed silver, not human expert gold.

## Completion and QC

- returned/schema-valid rows: 54/54
- direct exact evidence rows: 52
- explained missing-evidence rows: 1
- exact-substring evidence repairs: 1
- new resolved model-silver rows: 47
- new review-only rows: 7
- all required adjudications completed: 70/70
- all required resolved model-silver rows: 62
- all required review-only rows: 8
- returned failure buckets: hidden_or_missing_failure=1, no_failure=33, unclear=7, visible_failure=13
- returned final scores: 1=2, 2=7, 3=8, 4=19, 5=18

One returned evidence field was a faithful paraphrase rather than an exact
source substring. It was replaced with an exact sentence from the same
evaluator output. Its score, range, failure bucket, confidence, and training
use were not changed.

## Interpretation

The source-disagreement table compares existing labels and teacher outputs
to the model-review silver result only. It must not be presented as accuracy
against human expert gold. Unclear or low-confidence rows remain review_only
with zero downstream quality weight.

## Decision

Exp27N passes its completion and evidence-QC gates. The next permitted step
is construction of the controlled 361-row in-place downstream pilot variants.
This does not authorize model training yet, full-3326 expansion, or dev/test
relabeling.
