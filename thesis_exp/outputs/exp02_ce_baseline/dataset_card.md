# Exp2 CE Baseline Dataset

Exp2 uses the locked Exp0.1 paper-like triple split and converts it into a
5-class sequence-classification dataset.

The baseline input is question + answer + metric only. Rubric-aware and
metadata-aware inputs are reserved for Exp3.

Generated files:

- `data/train.jsonl`
- `data/dev.jsonl`
- `data/test.jsonl`

Expected split sizes:

| split | rows |
| --- | ---: |
| train | 2654 |
| dev | 664 |
| test | 2218 |
