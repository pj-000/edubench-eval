# Exp27O 361-Row In-Place Pilot Dataset Preparation

Exp27O keeps the same 3,326-row question-disjoint train universe in every
variant. The existing audit changes supervision only for 361 train rows;
the remaining 2,965 rows retain their original human labels.

## Variants

- `v0_original_unweighted`: original benchmark labels and unit weights.
- `v1_original_label_matched_weight`: original labels with the locked audit weights.
- `v2_selective_hard_relabel`: hard labels derived from the locked audit policy.
- `v3_selective_soft_audit`: tier-specific soft targets and confidence weights.

The extra v1 weight-only ablation separates the effect of sample weighting
from the effect of teacher/adjudication targets.

## Locked counts

- unaudited original rows: 2,965
- direct accept: 173
- weighted accept: 118
- adjudication required: 70
- resolved model-silver adjudications: 62
- review-only rows with zero weight: 8
- active-weight normalization factor: 1.02431118
- changed hard labels: 109
- upward/downward changes: 84/25
- original low labels changed to hard 4/5: 16

## Protocol

Full JSONL datasets remain private/local because they repeat original answer
text. Public artifacts contain only aggregate summaries and SHA-256 hashes.
No teacher API, GPU, training, dev label, or test label was used.

## High-impact transition gate

Sixteen original label-1/2 rows become hard label 4/5 under resolved
model-review silver. This may reflect genuine correction of noisy original
low labels, but it is also the highest-impact case group for the low-to-high
research objective. The pilot must report this group separately. No new API
annotation is required before the controlled pilot.

## Decision

Dataset construction passes. The next step is implementation and preflight
validation of one shared Qwen3-Reranker-0.6B pilot trainer that supports
per-example weights and five-class soft targets. Training remains gated until
that workflow passes review and smoke tests; the 16 high-impact upward
transitions remain a mandatory diagnostic slice.
