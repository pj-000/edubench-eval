# Exp16A-Stability Summary

This report summarizes qmr and metric_rubric dev stability across available seeds. Missing seeds are
warnings, not failures.

Requested seeds: `42, 43, 44`.
Available rows: `6`.

## Aggregate

| variant | n seeds | mean MAE | std MAE | mean QWK | std QWK | mean low-to-high | mean label2 recall | monotonic zero |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `qmr` | 3 | 0.4342 | 0.0308 | 0.4718 | 0.0728 | 0.6316 | 0.0000 | True |
| `metric_rubric` | 3 | 0.4435 | 0.0250 | 0.4674 | 0.0532 | 0.6491 | 0.0000 | True |

## Interpretation

- qmr has lower available-seed dev MAE than metric_rubric.
- qmr has lower available-seed low-to-high rate.
- label2 recall remains zero for both variants in the available seeds.

## RQ1 Exit Condition

RQ1 can be treated as ready to close only after the boundary diagnostic and at least the
available-seed stability summary support a consistent boundary-generation story. If seed 43/44 are
missing, this report should be read as a seed42 snapshot rather than a stability result.
