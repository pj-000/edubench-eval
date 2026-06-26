# Exp16A-Stability Summary

This report summarizes qmr and metric_rubric dev stability across available seeds. Missing seeds are
warnings, not failures.

Requested seeds: `42, 43, 44`.
Available rows: `2`.

## Aggregate

| variant | n seeds | mean MAE | std MAE | mean QWK | std QWK | mean low-to-high | mean label2 recall | monotonic zero |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `qmr` | 1 | 0.3993 | 0.0000 | 0.5532 | 0.0000 | 0.5088 | 0.0000 | True |
| `metric_rubric` | 1 | 0.4237 | 0.0000 | 0.5274 | 0.0000 | 0.5789 | 0.0000 | True |

## Interpretation

- Multi-seed stability is not yet established; run seed 43/44 before making a stability claim.
- qmr has lower available-seed dev MAE than metric_rubric.
- qmr has lower available-seed low-to-high rate.
- label2 recall remains zero for both variants in the available seeds.

## RQ1 Exit Condition

RQ1 can be treated as ready to close only after the boundary diagnostic and at least the
available-seed stability summary support a consistent boundary-generation story. If seed 43/44 are
missing, this report should be read as a seed42 snapshot rather than a stability result.
