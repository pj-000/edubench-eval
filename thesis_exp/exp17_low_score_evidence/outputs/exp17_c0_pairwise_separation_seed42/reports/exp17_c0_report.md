# Exp17-C0 Pairwise-Low Quality Separation Report

C0 keeps Exp16A qmr Boundary Linking and adds a pairwise separation loss on `quality_score_s` only.

## Guardrails

- Test split is not read.
- Dev D1 annotations are used only for evaluation.
- Human rationale text is not used as ranker input.
- Pairwise loss does not alter tau directly and does not use scalar `h`.

## Completed Configs

| config | pair source | MAE | QWK | low-to-high | label2 recall | mean g_i3 label2 | D1 s gap | D1 s AUC | train pair gap | train pairs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `C0_0_ordinal_continue` | `none` | 0.3902 | 0.5621 | 0.5439 | 0.0000 | 2.0160 | -0.3790 | 0.3102 | NA | 0 |
| `C0_10_exclude_format_auxiliary_gamma0p05_m0p2` | `exclude_format_auxiliary` | 0.3875 | 0.5719 | 0.4912 | 0.0000 | 2.1672 | -0.6090 | 0.2563 | 5.3715 | 400 |
| `C0_11_exclude_answer_key_dependent_gamma0p05_m0p2` | `exclude_answer_key_dependent` | 0.3830 | 0.5690 | 0.4912 | 0.0000 | 2.4246 | -0.5760 | 0.2485 | 4.8758 | 400 |
| `C0_12_random_matched_metric_rubric_gamma0p05_m0p2` | `random_matched_metric_rubric` | 0.3902 | 0.5766 | 0.5088 | 0.0789 | 2.1298 | -0.4856 | 0.3139 | 4.3256 | 420 |
| `C0_13_random_matched_metric_rubric_subject_gamma0p05_m0p2` | `random_matched_metric_rubric_subject` | 0.4002 | 0.5502 | 0.5088 | 0.0000 | 2.0471 | -0.1407 | 0.3866 | 3.6229 | 216 |
| `C0_14_same_question_group_upper_bound_gamma0p05_m0p2` | `same_question_group_upper_bound` | 0.3930 | 0.5570 | 0.5088 | 0.0000 | 2.0028 | -0.2282 | 0.3724 | 2.9708 | 123 |
| `C0_2_all_pairs_gamma0p05_m0p2` | `all_a0_pairs` | 0.3884 | 0.5688 | 0.4912 | 0.0000 | 2.9521 | -0.4135 | 0.3322 | 4.2338 | 420 |
| `C0_5_evidence_positive_plus_pairwise_low_gamma0p05_m0p2` | `evidence_positive_plus_pairwise_low` | 0.3875 | 0.5699 | 0.4912 | 0.0000 | 2.1825 | -0.6890 | 0.2279 | 5.0080 | 400 |
| `C0_6_random_pair_control_gamma0p05_m0p2` | `random_low_high_pairs` | 0.3866 | 0.5795 | 0.4912 | 0.0000 | 2.7168 | -0.2393 | 0.3763 | 4.8354 | 420 |
| `C0_7_same_subject_only_gamma0p05_m0p2` | `same_subject_only` | 0.4011 | 0.5635 | 0.5263 | 0.0000 | 2.5536 | -0.2725 | 0.3576 | 4.3374 | 216 |
| `C0_8_high_weight_only_gamma0p05_m0p2` | `high_weight_only_p75` | 0.4029 | 0.5770 | 0.4912 | 0.0000 | 2.3934 | -0.4329 | 0.2732 | 4.0106 | 106 |
| `C0_9_same_subject_high_weight_gamma0p05_m0p2` | `same_subject_high_weight_p75` | 0.3939 | 0.5714 | 0.5088 | 0.0000 | 2.4368 | -0.6317 | 0.2979 | 4.3863 | 73 |

## Noise-Control Comparisons

| comparison | left best l2h | right best l2h | left D1 AUC | right D1 AUC |
|---|---:|---:|---:|---:|
| `all_a0_pairs` vs `same_subject_only` | 0.4912 | 0.5263 | 0.3322 | 0.3576 |
| `all_a0_pairs` vs `high_weight_only_p75` | 0.4912 | 0.4912 | 0.3322 | 0.2732 |
| `all_a0_pairs` vs `same_subject_high_weight_p75` | 0.4912 | 0.5088 | 0.3322 | 0.2979 |
| `all_a0_pairs` vs `random_low_high_pairs` | 0.4912 | 0.4912 | 0.3322 | 0.3763 |
| `all_a0_pairs` vs `random_matched_metric_rubric` | 0.4912 | 0.5088 | 0.3322 | 0.3139 |
| `all_a0_pairs` vs `same_question_group_upper_bound` | 0.4912 | 0.5088 | 0.3322 | 0.3724 |

## Decision

- final_decision: `C0_success_risk`
- best_main_config: `C0_2_all_pairs_gamma0p05_m0p2`
- best_upper_bound_config: `C0_14_same_question_group_upper_bound_gamma0p05_m0p2`
- latent_success: `True`
- risk_success: `True`
- main_method_success: `True`
- c0_success: `True`
- enter_c1: `True`
- enter_b1: `False`
- low_to_high_delta_vs_c0_0: -0.0526
- mean_g_i3_label2_delta_vs_c0_0: 0.9361
- dev_d1_s_gap_delta_vs_c0_0: -0.0345
- A0 pairs outperform random matched metric/rubric control: `True`
- upper_bound_diagnostic_success: `True`

## Decision Questions

- Did any main C0 config achieve risk success? `True`
- Did any main C0 config achieve latent success? `True`
- Did same_question_group_upper_bound only succeed? `False`
- Are A0 pairs better than random matched metric/rubric controls? `True`
- Should we proceed to C1? `True`
- Should we still block B1 suppression? `True`

C0 must pass the success gate before any B1 suppression experiment.
