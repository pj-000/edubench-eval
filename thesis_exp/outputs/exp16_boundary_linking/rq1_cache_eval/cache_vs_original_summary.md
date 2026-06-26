# Exp16A Boundary Cache vs Original Diagnosis

Original diagnosis: `thesis_exp/outputs/exp16_boundary_linking/rq1_diagnosis`.
Boundary-cache diagnosis: `thesis_exp/outputs/exp16_boundary_linking/rq1_cache_eval/rq1_diagnosis`.

Boundary cache is an eval/export-only diagnostic. It does not change training, loss, or checkpoint
selection.

## Summary

| variant | split | low-to-high original -> cache | label2 recall original -> cache | unstable groups original -> cache | p95 max tau diff original -> cache | max tau diff original -> cache |
|---|---|---:|---:|---:|---:|---:|
| `metric_rubric` | dev | 0.5789 -> 0.5789 | 0.0000 -> 0.0000 | 16 -> 0 | 0.017705 -> 0.000000 | 0.026184 -> 0.000000 |
| `qmr` | dev | 0.5088 -> 0.5088 | 0.0000 -> 0.0000 | 99 -> 0 | 0.031530 -> 0.000000 | 0.058834 -> 0.000000 |

## Conclusion

Boundary cache does not change the main RQ1 conclusion: pointwise boundary linking still fails to
recover label-2 answers, so the remaining failure is not explained only by repeated boundary
encoding noise.
If cache reduces boundary_key tau variation to near zero while label2 recall remains zero and
low-to-high remains high, the evidence still supports the interpretation that quality_score_s does
not separate label-2 answers from middle/high-score answers well enough.
