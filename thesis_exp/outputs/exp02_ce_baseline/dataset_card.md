# Exp2 CE Baseline Dataset

This dataset is derived from the locked Exp0.1 paper-like triple split. It is intended for
training a 5-class cross-entropy baseline for EduBenchEvaluator 0.6B.

No Exp0 data or split file is modified. Human labels are used only as supervised targets;
existing automatic judge predictions are not used as training targets.

Exp2 baseline input = question + answer + metric only.
Rubric-aware / metadata-aware inputs are reserved for Exp3.

## Source Splits

| split | source | rows | expected_rows | status |
| --- | --- | ---: | ---: | --- |
| train | `thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl` | 2654 | 2654 | PASS |
| dev | `thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl` | 664 | 664 | PASS |
| test | `thesis_exp/data/splits/paper_like_triple_seed42/test.jsonl` | 2218 | 2218 | PASS |

## Label Distribution

| split | label_5 | count | pct_within_split |
| --- | ---: | ---: | ---: |
| train | 1 | 24 | 0.009043 |
| train | 2 | 52 | 0.019593 |
| train | 3 | 251 | 0.094574 |
| train | 4 | 946 | 0.356443 |
| train | 5 | 1381 | 0.520347 |
| dev | 1 | 6 | 0.009036 |
| dev | 2 | 14 | 0.021084 |
| dev | 3 | 62 | 0.093373 |
| dev | 4 | 237 | 0.356928 |
| dev | 5 | 345 | 0.519578 |
| test | 1 | 56 | 0.025248 |
| test | 2 | 47 | 0.021190 |
| test | 3 | 194 | 0.087466 |
| test | 4 | 720 | 0.324617 |
| test | 5 | 1201 | 0.541479 |

## Generated Files

- `thesis_exp/outputs/exp02_ce_baseline/data/train.jsonl`
- `thesis_exp/outputs/exp02_ce_baseline/data/dev.jsonl`
- `thesis_exp/outputs/exp02_ce_baseline/data/test.jsonl`
