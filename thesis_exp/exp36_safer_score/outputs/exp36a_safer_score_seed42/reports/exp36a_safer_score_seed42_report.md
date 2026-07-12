# Exp36A SAFER-Score Seed42 Report

## Locked scope

Train-only OOF supervision and one dev-only seed42 scout. No API and no test access.

## Selected metrics

| Variant | Epoch | MAE | Exact | QWK | L2H | Label2 recall | Label5 recall | H2L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0_original_hard | 5 | 0.3434 | 0.7048 | 0.5926 | 0.5000 | 0.2143 | 0.8261 | 0.0017 |
| v0h_human_soft | 7 | 0.3117 | 0.7244 | 0.6539 | 0.5000 | 0.2143 | 0.8116 | 0.0017 |
| v1_qwen_hard | 1 | 0.5873 | 0.5422 | 0.0916 | 1.0000 | 0.0000 | 0.9072 | 0.0000 |
| v2_qwen_range_soft | 1 | 0.5904 | 0.5392 | 0.0879 | 1.0000 | 0.0000 | 0.9014 | 0.0000 |
| v3_naive_human_qwen | 4 | 0.3358 | 0.7033 | 0.6086 | 0.4500 | 0.3571 | 0.7768 | 0.0017 |
| v4_human_soft_logit_adjustment | 8 | 0.4172 | 0.6732 | 0.5739 | 0.2500 | 0.5000 | 0.7507 | 0.0378 |
| v5_safer_score | 10 | 0.3223 | 0.7154 | 0.6564 | 0.4000 | 0.2143 | 0.7797 | 0.0034 |
| v7_shuffled_teacher_control | 7 | 0.3223 | 0.7169 | 0.6528 | 0.4500 | 0.2143 | 0.7884 | 0.0052 |

## Gate

Strong baseline: `v0h_human_soft`.

- mae_guard: FAIL
- exact_guard: PASS
- qwk_guard: PASS
- label5_guard: FAIL
- high_to_low_guard: PASS
- low_tail_improvement: PASS
- beats_naive_or_shuffle: PASS
- no_nan_or_oom: PASS
- no_test_access: PASS

## Supervision audit

- OOF coverage: 2654/2654; missing=0; duplicate=0; question-key leakage=0.
- Human labels 1/2/3/4/5: 24/52/251/946/1381.
- Mean normalized human entropy: 0.2213.
- Qwen explicit score-range coverage: 0.0000; all 2654 rows use hard-score fallback.
- Evidence-gate pass: 2611/2654 (0.9838).
- DeepSeek coverage: 1552/2654 (0.5848); score-confirmed=1376.
- V5 failure-head masked rows: 2429; train micro-F1=0.8620; exact-set-match=0.8584.

## Boundary diagnosis

- V5 minus strong baseline MAE: +0.010542; allowed maximum was +0.010000.
- V5 minus strong baseline Exact: -0.009036.
- V5 minus strong baseline QWK: +0.002547.
- V5 minus strong baseline low-to-high: -0.100000.
- V5 minus strong baseline label5 recall: -0.031884; allowed minimum was -0.030000.
- V5 versus shuffled V7: MAE delta=+0.000000, Exact delta=-0.001506, low-to-high delta=-0.050000.
- The semantic teacher alignment shows a weak low-risk signal, but V7 remains too close and the preregistered accuracy/high-score guards fail.
- The dev low tail has only 20 rows, so risk estimates are uncertain; the paired question-key bootstrap table must accompany any claim.

Decision: **STOP_SAFER_SCORE**.

recommend_run_seeds_43_44 = `false`
stop_safer_score = `true`

No API, no test access, and no heavy/private artifacts are intended for commit.
