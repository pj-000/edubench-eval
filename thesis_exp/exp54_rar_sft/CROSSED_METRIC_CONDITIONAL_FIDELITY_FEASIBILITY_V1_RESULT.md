# Crossed-Metric Conditional-Fidelity Feasibility V1

## Decision

`NO_GO_INSUFFICIENT_CROSSED_METRIC_SUPPORT`

The frozen dev split contains 131 QA nodes evaluated on at least two metrics,
which passes the node-count requirement of 80. However, those nodes are spread
over sparse metric combinations. The largest dev metric pair shares only 11 QA
nodes, no dev metric pair reaches the preregistered requirement of 20, and zero
metric pairs jointly satisfy the train >=30 and dev >=20 requirements. The
required number of jointly eligible pairs was six.

Consequently, the full crossed-metric conditional-fidelity audit is not
identified on the current dev split. Human contrast prevalence, leave-one-rater
metric concordance, trait compression, and marginal-conditional gain gaps are
not estimated by pooling sparse pairs after observing this failure.

This result does **not** show that rubric conditional insensitivity is absent.
It shows that the current dev design cannot distinguish that mechanism reliably
under the proposed within-QA, within-metric-pair design.

## Verified inventory

| Quantity | Train | Dev |
|---|---:|---:|
| Rows | 2,654 | 664 |
| Unique QA nodes | 906 | 502 |
| Multi-metric QA nodes | 743 | 131 |
| Maximum shared QA for one metric pair | 176 | 11 |
| Metric pairs reaching the split threshold | 41 (>=30) | 0 (>=20) |
| Duplicate response-metric edges | 16 | 5 |

Metric-pair counts deduplicate repeated response-metric edges, so duplicate rows
cannot inflate support.

## Scientific consequence

No new loss, preference objective, reasoning module, or rubric-conditioned
method is authorized from this diagnostic. The current RAR-SFT and Field-DPO
results remain suitable for the master's thesis, with the lack of crossed-metric
confirmation recorded as a split-design limitation. A future conditional-
fidelity study would need a deliberately crossed, response-disjoint validation
set rather than additional analysis of the current dev split.

No test data, GPU, model inference, API call, or new training was used.
