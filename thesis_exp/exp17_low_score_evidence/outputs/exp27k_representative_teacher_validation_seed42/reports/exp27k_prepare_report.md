# Exp27K Representative Teacher Validation Preparation

Exp27K fills missing Qwen/DeepSeek coverage in the Exp27J representative probability sample.
It does not train and does not expose the Exp27J silver reference to either teacher.

## Coverage

- representative rows: 120
- prior teacher-covered rows: 47
- missing rows prepared for API audit: 73

## Protocol

- blind stage input: question context, evaluator output, metric, rubric, and metadata only.
- label-aware audit stage additionally sees only the original train human score.
- Exp27J reviewer scores, adjudications, failure buckets, and final silver scores are excluded.
- Exp27K does not open dev/test; it inherits Exp27J's already-verified zero-overlap audit.

## Gate

Formal downstream training remains blocked until all missing API outputs are complete and protocol
validation is rerun.
