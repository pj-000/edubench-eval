# QD-PR2 Anchored Pairwise Ordinal Fine-tuning Result

Run ID: `QD-PR2_AnchoredPairwiseOrdinal_human_only`
Formal training status: `completed`.
Training executed: `yes`, on server.
API called: `no`.
Synthetic generated: `no`.
Checkpoint/weights submitted: `no`.

## Setup

QD-PR2 initializes from the QD-B1 checkpoint and performs small-weight pairwise fine-tuning rather
than from-scratch training. The formal run used question_seed42 human-only A4 data, no synthetic rows,
`lambda_pair=0.05`, `lambda_anchor=0.5`, `lambda_mono=0.1`, `epochs=3`, and effective batch size `128`.

## Test Result vs QD-B1

| Model | low_to_high | MAE | QWK | Acc@5 | high_to_mid_or_low | signed_bias | monotonic_violation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QD-B1_human_only_L1_weighted_ordinal | 0.4516 | 0.4279 | 0.6012 | 0.7419 | 0.0146 | 0.0181 | 0.3119 |
| QD-PR2_AnchoredPairwiseOrdinal_human_only | 0.3871 | 0.4192 | 0.6084 | 0.7549 | 0.0162 | 0.0335 | 0.3527 |

## Interpretation

- Low-to-high improved vs QD-B1: `TRUE` (`0.4516` -> `0.3871`).
- MAE improved vs QD-B1: `TRUE` (`0.4279` -> `0.4192`).
- QWK improved vs QD-B1: `TRUE` (`0.6012` -> `0.6084`).
- Acc@5 improved vs QD-B1: `TRUE` (`0.7419` -> `0.7549`).
- Monotonicity worsened vs QD-B1: `TRUE` (`0.3119` -> `0.3527`).

QD-PR2 is a clear improvement over QD-B1 on the primary low-to-high error and also improves MAE/QWK,
but it does not preserve ordinal monotonicity. This should be reported as a promising anchored pairwise
fine-tuning result with a remaining monotonicity defect, not as a fully solved final scorer.

## Pairwise Diagnostics

| Pair diagnostic | value |
| --- | ---: |
| dev pairwise accuracy overall | 0.8040 |
| dev margin satisfied overall | 0.5557 |
| dev low_high pairwise accuracy | 0.7933 |
| dev low_high margin satisfied | 0.4883 |

## Raw Outputs

Raw predictions, arrays, and full logs are available locally under
`thesis_exp/outputs/exp09_pairwise_ordinal_qdpr2/runs/QD-PR2_AnchoredPairwiseOrdinal_human_only/`.
These raw run files remain Git-ignored to avoid committing large predictions/arrays. The tracked summary tables are:

- `tables/qdpr2_formal_metrics_summary.csv`
- `tables/qdpr2_delta_vs_qdb1.csv`

## Next Step

Run diagnosis before deciding whether to continue. The main question is why anchored pairwise fine-tuning
still leaves `monotonic_violation_rate=0.3527` despite the added monotonic regularization.
