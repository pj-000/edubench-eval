# Exp17-C0 Pair Noise Audit

This audit reads train-side pair metadata only. It does not train a model and does not read test.

## Summary

- n_pairs: 420
- same_metric_rate: 1.0000
- same_language_rate: 1.0000
- same_rubric_hash_rate: 1.0000
- same_subject_rate: 0.5143
- same_question_group_rate: 0.0000
- same_boundary_key_rate: 0.0000
- pair_weight_p50: 0.5712
- pair_weight_p75: 0.6434
- pair_weight_p90: 0.6643

## Interpretation

- Current pairs are mainly cross-question pairs if same_question_group_rate is near 0.
- same_subject controls whether cross-subject noise is likely to enter the pairwise signal.
- high_weight_only_p75 tests whether A0 pair confidence helps filter noisy preferences.
- strict filters with enough pairs: all_a0_pairs, same_subject_only, high_weight_only_p75, same_subject_high_weight_p75, exclude_format_auxiliary, exclude_answer_key_dependent, weak_evidence_only, missing_key_point_only, insufficient_evidence_only, random_matched_metric_rubric, random_matched_metric_rubric_subject, same_question_group_upper_bound
- unavailable or too small pair sources: factual_or_rubric_mismatch_only
- recommended_direct_all_pairs_only: no; run all pairs only as a preliminary baseline with noise-control ablations

## Recommended C0 Scout Configs

- C0_0_ordinal_continue
- C0_1/C0_2/C0_3 all_a0_pairs
- C0_7 same_subject_only
- C0_8 high_weight_only_p75
- C0_9 same_subject_high_weight_p75
- C0_10 exclude_format_auxiliary
- C0_11 exclude_answer_key_dependent
- C0_12 random_matched_metric_rubric
- C0_13 random_matched_metric_rubric_subject
- C0_14 same_question_group_upper_bound only as an upper-bound diagnostic if available
