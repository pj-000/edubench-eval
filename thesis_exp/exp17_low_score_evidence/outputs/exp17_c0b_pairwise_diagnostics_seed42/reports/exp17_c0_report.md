# Exp17-C0 Pairwise-Low Quality Separation Report

C0 keeps Exp16A qmr Boundary Linking and adds a pairwise separation loss on `quality_score_s` only.

## Guardrails

- Test split is not read.
- Dev D1 annotations are used only for evaluation.
- Human rationale text is not used as ranker input.
- Pairwise loss does not alter tau directly and does not use scalar `h`.

## Completed Configs

| config | pair source | loss space | freeze boundary | MAE | QWK | low-to-high | label2 recall | mean g_i3 label2 | D1 s gap | D1 s AUC | train pair gap | train pairs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `C0b_0_init_no_train_eval` | `none` | `raw_s` | `False` | 0.3993 | 0.5532 | 0.5088 | 0.0000 | 2.2316 | -0.0365 | 0.3985 | NA | 0 |
| `C0b_1_all_pairs_raw_s_gamma0p01_freeze_boundary` | `all_a0_pairs` | `raw_s` | `True` | 0.3948 | 0.5608 | 0.5088 | 0.0000 | 1.8453 | -0.4711 | 0.2993 | 6.9824 | 420 |
| `C0b_2_all_pairs_raw_s_gamma0p02_freeze_boundary` | `all_a0_pairs` | `raw_s` | `True` | 0.3921 | 0.5625 | 0.5088 | 0.0000 | 1.8388 | -0.4752 | 0.2968 | 7.0576 | 420 |
| `C0b_3_evidence_pairs_raw_s_gamma0p02_freeze_boundary` | `evidence_positive_plus_pairwise_low` | `raw_s` | `True` | 0.3884 | 0.5679 | 0.5088 | 0.0000 | 1.8542 | -0.5123 | 0.2798 | 7.4704 | 400 |
| `C0b_4_random_pair_raw_s_gamma0p02_freeze_boundary` | `random_low_high_pairs` | `raw_s` | `True` | 0.3884 | 0.5697 | 0.5088 | 0.0000 | 1.7799 | -0.5368 | 0.2984 | 7.5122 | 420 |
| `C0b_5_all_pairs_g3detach_gamma0p01` | `all_a0_pairs` | `g3_detached` | `False` | 0.3857 | 0.5839 | 0.4912 | 0.0000 | 2.7808 | -0.3248 | 0.3541 | 4.0617 | 420 |
| `C0b_6_all_pairs_g3detach_gamma0p02` | `all_a0_pairs` | `g3_detached` | `False` | 0.3866 | 0.5764 | 0.4912 | 0.0000 | 3.0147 | -0.3773 | 0.3358 | 4.1118 | 420 |
| `C0b_7_random_matched_g3detach_gamma0p02` | `random_matched_metric_rubric` | `g3_detached` | `False` | 0.3902 | 0.5760 | 0.4912 | 0.0000 | 2.7052 | -0.4594 | 0.3171 | 3.7466 | 420 |

## Noise-Control Comparisons

| comparison | left best l2h | right best l2h | left D1 AUC | right D1 AUC |
|---|---:|---:|---:|---:|
| `all_a0_pairs` vs `random_low_high_pairs` | 0.4912 | 0.5088 | 0.3541 | 0.2984 |
| `all_a0_pairs` vs `random_matched_metric_rubric` | 0.4912 | 0.4912 | 0.3541 | 0.3171 |

## Decision

- final_decision: `C0_latent_success_only`
- baseline_config: `C0b_0_init_no_train_eval`
- best_main_config: `C0b_1_all_pairs_raw_s_gamma0p01_freeze_boundary`
- best_upper_bound_config: ``
- latent_success: `True`
- risk_success: `False`
- main_method_success: `False`
- c0_success: `False`
- enter_c1: `False`
- enter_b1: `False`
- low_to_high_delta_vs_c0_0: 0.0000
- mean_g_i3_label2_delta_vs_c0_0: -0.3863
- dev_d1_s_gap_delta_vs_c0_0: -0.4346
- A0 pairs outperform random matched metric/rubric control: `True`
- upper_bound_diagnostic_success: `False`

## Decision Questions

- Did any main C0 config achieve risk success? `False`
- Did any main C0 config achieve latent success? `True`
- Did same_question_group_upper_bound only succeed? `False`
- Are A0 pairs better than random matched metric/rubric controls? `True`
- Should we proceed to C1? `False`
- Should we still block B1 suppression? `True`

C0 must pass the success gate before any B1 suppression experiment.
