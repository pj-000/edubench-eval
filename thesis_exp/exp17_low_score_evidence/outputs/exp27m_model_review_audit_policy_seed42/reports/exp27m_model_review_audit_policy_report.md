# Exp27M Model-Review Audit Policy Acceptance

Exp27M is a CPU-only, train-only acceptance audit. Reviewer A/B and the
third adjudicator are model reviewers. The resulting reference is silver,
not human expert gold, and no row is emitted as trainer data.

## Completion

- unique reviewed samples: 107
- packet appearances: 114
- required/completed adjudications: 38/38
- API calls: 0
- GPU runs: 0
- model training runs: 0

## Balanced-60 source comparison

The packet contains 60 rows; 58 are quantitative and 2 unresolved rows are qualitative-only.

- original_human: MAE=0.8621, QWK=0.6489, within-one=0.7759, severe=0.2241
- qwen_only: MAE=0.4310, QWK=0.8067, within-one=0.9138, severe=0.0862
- deepseek_only: MAE=0.5862, QWK=0.7441, within-one=0.8621, severe=0.1379
- dual_teacher_half_up_mean: MAE=0.4310, QWK=0.8356, within-one=0.8966, severe=0.1034
- human_qwen_deepseek_median: MAE=0.4828, QWK=0.8095, within-one=0.8793, severe=0.1207
- exp27i_v1: MAE=0.4828, QWK=0.8095, within-one=0.8793, severe=0.1207
- oof_soft_fusion_nll_expected: MAE=0.5000, QWK=0.8133, within-one=0.9310, severe=0.0690
- oof_soft_fusion_rps_expected: MAE=0.5000, QWK=0.8133, within-one=0.9310, severe=0.0690

## Selected policy

- selected signal: `qwen_human_gap`
- review budget: 0.20
- policy MAE with selective model review: 0.2931
- best simple source: `dual_teacher_half_up_mean` (MAE=0.4310)
- review-tier predictions use the completed model-review result, so their accuracy includes review cost and is not a raw-source comparison.

## Tier reliability

- direct_accept: n=24, coverage=0.4138, within-one=0.9583333333333334, severe=0.041666666666666664
- weighted_accept: n=23, coverage=0.3966, within-one=0.8260869565217391, severe=0.17391304347826086
- adjudication_required: n=11, coverage=0.1897, within-one=0.2727272727272727, severe=0.7272727272727273

## High-control lockbox

- rows: 34
- hard conflict rate: 0.0000
- used for policy selection: false

## Acceptance checks

- PASS: all_107_unique_reviews_merged
- PASS: all_required_adjudications_completed
- PASS: selected_policy_exists
- PASS: direct_accept_within_one_ge_0p90
- PASS: direct_accept_severe_error_le_0p10
- PASS: direct_accept_cluster_bootstrap_upper_le_0p20
- PASS: review_error_at_least_2x_direct
- PASS: review_budget_le_0p20
- PASS: high_control_hard_conflict_le_0p10
- PASS: policy_MAE_within_best_simple_plus_0p05
- PASS: model_review_marked_silver
- PASS: unclear_balanced_rows_excluded_from_quantitative_selection

## Decision

- status: `PASS`
- proceed to 361-row in-place downstream pilot: true
- proceed to full 3326 expansion: false
- proceed to formal Qwen3-Reranker training: false
- relabel dev/test: false

The next allowed experiment is a controlled 361-row in-place downstream pilot.
The same 3326-row universe, exclusions, effective weights, optimizer, and
checkpoint rule must be shared across its ablations.
