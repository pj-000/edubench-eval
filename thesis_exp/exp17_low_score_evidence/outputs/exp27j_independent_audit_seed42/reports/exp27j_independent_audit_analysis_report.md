# Exp27J Independent Audit Analysis

## Completion

- reviewer A: 180/180
- reviewer B: 180/180
- required adjudications: 68
- completed required adjudications: 68
- final reference rows: 180/180

## Reviewer Reliability

- QWK: 0.901343
- ordinal alpha (quadratic disagreement): 0.901493
- reference status: dual_codex_blind_review_model_adjudicated_silver_reference

## Exp27I Implementation Audit

- independent top80 adjudication input found: False
- current top80 is rule-based: True
- current 292 formal-training-ready: False

## Decision

- proceed to Qwen3-Reranker downstream experiment: False
- blocking criteria: ['review_only_error_rate_gt_high_weight', 'tier_trust_ordering_high_gt_low_gt_review']
- recommendation: revise_exp27i_calibration_tiers_using_exp27j_then_external_review
- representative prevalence uses design weights and question-key cluster bootstrap.
- Teacher/Exp27I metrics on the representative view use only rows with existing Exp27I coverage; coverage is reported and missing rows are never treated as non-conflicts.
- risk-enriched metrics are stress-test results and are not population prevalence estimates.
- Alternative Annotator Test input is prepared; no unverified implementation was run.

## Source Scores Against Silver Reference

- original_label_5: n=180/180, coverage=1.000000, MAE=1.066667, QWK=0.405485, within-one=0.694444, low-to-high=0.250000
- human_mean: n=180/180, coverage=1.000000, MAE=1.066667, QWK=0.405485, within-one=0.694444, low-to-high=0.250000
- human_median: n=180/180, coverage=1.000000, MAE=1.066667, QWK=0.405485, within-one=0.694444, low-to-high=0.250000
- qwen: n=107/180, coverage=0.594444, MAE=0.644860, QWK=0.678576, within-one=0.869159, low-to-high=0.065217
- deepseek: n=107/180, coverage=0.594444, MAE=0.738318, QWK=0.681446, within-one=0.841121, low-to-high=0.043478
- naive_human_qwen_deepseek_median: n=107/180, coverage=0.594444, MAE=0.607477, QWK=0.700795, within-one=0.878505, low-to-high=0.065217
- exp27i_calibrated: n=107/180, coverage=0.594444, MAE=0.607477, QWK=0.700795, within-one=0.878505, low-to-high=0.065217

## Exp27I Tier Validation

- high_weight: n=46, MAE=0.282609, within-one=0.934783, abs>=2 error=0.065217
- low_weight: n=13, MAE=0.923077, within-one=0.769231, abs>=2 error=0.230769
- review_only: n=48, MAE=0.833333, within-one=0.854167, abs>=2 error=0.145833
- review-only minus high-weight abs>=2 error: estimate=0.080616, 95% CI=[-0.078405, 0.241379]

## Representative View

- original_label_conflict_abs_ge_2: estimate=0.180046, 95% CI=[0.104990, 0.260111], coverage=120/120 (population estimate)
- teacher_human_conflict_abs_ge_2: estimate=0.146311, 95% CI=[0.060252, 0.300308], coverage=47/120 (observed covered subset only)
- evidence_gap_or_hidden_failure: estimate=0.099479, 95% CI=[0.044002, 0.164891], coverage=120/120 (population estimate)
- exp27i_review_only: estimate=0.112270, 95% CI=[0.046002, 0.232674], coverage=47/120 (observed covered subset only)

## Limitations

- The reference is a dual-Codex blind-review, model-adjudicated silver reference; no human domain expert participated.
- Original-label conflict rates mean disagreement with this silver reference, not proven human annotation error.
- The 180 rows contain 84 question-key clusters because train has only 118 unique question keys; uncertainty uses question-key cluster bootstrap.
- Only 47/120 representative rows have existing Qwen/DeepSeek/Exp27I coverage. Covered-subset teacher statistics are not full-population prevalence estimates.
- Current Exp27I top-80 wording overstates the implementation: the file is a generated rule-based output, not an external adjudication input.
