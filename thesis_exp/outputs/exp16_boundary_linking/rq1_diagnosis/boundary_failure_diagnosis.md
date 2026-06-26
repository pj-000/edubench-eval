# Exp16A-Diag Boundary Failure Diagnosis

This diagnostic uses saved prediction files only. It does not load checkpoints or train models.

## Analyzed Runs

- `metric_rubric` `dev`: `1107` rows
- `metric_rubric` `test`: `1103` rows
- `qmr` `dev`: `1107` rows
- `qmr` `test`: `1103` rows

## Per-Label Margin Summary

| variant | split | gold | n | mean s | mean tau3 | mean tau4 | mean margin tau3 | pred>=4 | recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `metric_rubric` | dev | 1 | 19 | -5.2296 | -0.7288 | 1.3081 | -4.5009 | 5/19 (0.2632) | 0.6842 |
| `metric_rubric` | dev | 2 | 38 | 1.8338 | 0.0847 | 1.9741 | 1.7492 | 28/38 (0.7368) | 0.0000 |
| `metric_rubric` | dev | 3 | 105 | 1.3661 | -0.2763 | 1.9384 | 1.6424 | 95/105 (0.9048) | 0.0952 |
| `metric_rubric` | dev | 4 | 389 | 1.4516 | -0.5681 | 1.7314 | 2.0197 | 381/389 (0.9794) | 0.6015 |
| `metric_rubric` | dev | 5 | 556 | 2.6711 | -0.9201 | 1.0099 | 3.5913 | 556/556 (1.0000) | 0.8381 |
| `metric_rubric` | test | 1 | 9 | -8.0842 | -1.3777 | 0.6115 | -6.7065 | 1/9 (0.1111) | 0.8889 |
| `metric_rubric` | test | 2 | 22 | 0.9178 | -0.0822 | 2.0016 | 1.0000 | 14/22 (0.6364) | 0.0000 |
| `metric_rubric` | test | 3 | 105 | 0.7647 | -0.6237 | 1.5888 | 1.3884 | 93/105 (0.8857) | 0.1143 |
| `metric_rubric` | test | 4 | 351 | 1.1372 | -0.8282 | 1.5029 | 1.9654 | 343/351 (0.9772) | 0.6325 |
| `metric_rubric` | test | 5 | 616 | 2.4394 | -1.0013 | 1.0872 | 3.4407 | 616/616 (1.0000) | 0.8133 |
| `qmr` | dev | 1 | 19 | -1.1340 | 0.4958 | 2.2944 | -1.6298 | 2/19 (0.1053) | 0.4737 |
| `qmr` | dev | 2 | 38 | 1.6080 | 0.0240 | 2.0229 | 1.5840 | 27/38 (0.7105) | 0.0000 |
| `qmr` | dev | 3 | 105 | 2.0532 | 0.3845 | 3.1403 | 1.6687 | 86/105 (0.8190) | 0.1810 |
| `qmr` | dev | 4 | 389 | 2.1364 | -0.1248 | 2.8318 | 2.2612 | 373/389 (0.9589) | 0.6838 |
| `qmr` | dev | 5 | 556 | 3.0024 | -0.9642 | 1.3449 | 3.9666 | 555/556 (0.9982) | 0.8183 |
| `qmr` | test | 1 | 9 | -2.7309 | -0.0686 | 1.3093 | -2.6623 | 1/9 (0.1111) | 0.8889 |
| `qmr` | test | 2 | 22 | 1.5447 | 0.2072 | 2.6922 | 1.3375 | 16/22 (0.7273) | 0.0000 |
| `qmr` | test | 3 | 105 | 1.5300 | 0.0823 | 2.8512 | 1.4477 | 87/105 (0.8286) | 0.1714 |
| `qmr` | test | 4 | 351 | 1.7746 | -0.4031 | 2.4882 | 2.1777 | 340/351 (0.9687) | 0.6695 |
| `qmr` | test | 5 | 616 | 2.8895 | -0.9508 | 1.4878 | 3.8403 | 611/616 (0.9919) | 0.7792 |

## Label-2 Failure Pattern

- `metric_rubric` `dev` label2 n=`38`, pred>=4=`28` (0.7368)；pred=1: 0, pred=2: 0, pred=3: 10, pred=4: 12, pred=5: 16.
- `metric_rubric` `test` label2 n=`22`, pred>=4=`14` (0.6364)；pred=1: 0, pred=2: 0, pred=3: 8, pred=4: 8, pred=5: 6.
- `qmr` `dev` label2 n=`38`, pred>=4=`27` (0.7105)；pred=1: 0, pred=2: 0, pred=3: 11, pred=4: 10, pred=5: 17.
- `qmr` `test` label2 n=`22`, pred>=4=`16` (0.7273)；pred=1: 0, pred=2: 0, pred=3: 6, pred=4: 9, pred=5: 7.

## Low-To-High Diagnosis

- `metric_rubric` `dev`: low-to-high `33`. mean s `2.1385`, mean tau3 `-0.2985`, mean tau4 `1.6151`, mean margin_tau3 `2.4370`. Diagnosis: `both` {'s_gap_l2h_minus_other_low': '6.3155', 'tau3_gap_l2h_minus_other_low': '-0.2660', 'margin_gap': '6.5815'}.
- `metric_rubric` `test`: low-to-high `15`. mean s `1.1951`, mean tau3 `-0.6275`, mean tau4 `1.3477`, mean margin_tau3 `1.8225`. Diagnosis: `both` {'s_gap_l2h_minus_other_low': '5.6008', 'tau3_gap_l2h_minus_other_low': '-0.3278', 'margin_gap': '5.9285'}.
- `qmr` `dev`: low-to-high `29`. mean s `1.9953`, mean tau3 `-0.6752`, mean tau4 `1.2149`, mean margin_tau3 `2.6705`. Diagnosis: `both` {'s_gap_l2h_minus_other_low': '2.6491', 'tau3_gap_l2h_minus_other_low': '-1.7436', 'margin_gap': '4.3926'}.
- `qmr` `test`: low-to-high `17`. mean s `1.9570`, mean tau3 `-0.1577`, mean tau4 `2.1687`, mean margin_tau3 `2.1148`. Diagnosis: `both` {'s_gap_l2h_minus_other_low': '3.6617', 'tau3_gap_l2h_minus_other_low': '-0.6307', 'margin_gap': '4.2924'}.

## Metric Concentration

| variant | split | metric | true low | low-to-high | label2 pred>=4 | mean low margin tau3 |
|---|---|---|---:|---:|---:|---:|
| `metric_rubric` | dev | Instruction Following & Task Completion | 7 | 7 (1.0000) | 7/7 (1.0000) | 3.3776 |
| `metric_rubric` | dev | Error Identification & Correction Precision | 5 | 5 (1.0000) | 4/4 (1.0000) | 3.2290 |
| `metric_rubric` | dev | Clarity, Simplicity & Inspiration | 4 | 4 (1.0000) | 4/4 (1.0000) | 2.3218 |
| `metric_rubric` | dev | Scenario Element Integration | 4 | 4 (1.0000) | 4/4 (1.0000) | 1.2772 |
| `metric_rubric` | dev | Basic Factual Accuracy | 2 | 2 (1.0000) | 2/2 (1.0000) | 4.7541 |
| `metric_rubric` | dev | Content Relevance & Scope Control | 2 | 2 (1.0000) | 2/2 (1.0000) | 4.4090 |
| `metric_rubric` | dev | Personalization, Adaptation & Learning Support | 1 | 1 (1.0000) | 1/1 (1.0000) | 0.1071 |
| `metric_rubric` | dev | Motivation, Guidance & Positive Feedback | 9 | 5 (0.5556) | 1/4 (0.2500) | 0.0486 |
| `metric_rubric` | dev | Reasoning Process Rigor | 16 | 3 (0.1875) | 3/3 (1.0000) | -5.3054 |
| `metric_rubric` | dev | Higher-Order Thinking & Skill Development | 7 | 0 (0.0000) | 0/7 (0.0000) | -1.0311 |
| `metric_rubric` | test | Instruction Following & Task Completion | 5 | 5 (1.0000) | 5/5 (1.0000) | 2.2841 |
| `metric_rubric` | test | Clarity, Simplicity & Inspiration | 2 | 2 (1.0000) | 2/2 (1.0000) | 1.2029 |
| `metric_rubric` | test | Basic Factual Accuracy | 1 | 1 (1.0000) | 1/1 (1.0000) | 4.0265 |
| `metric_rubric` | test | Content Relevance & Scope Control | 1 | 1 (1.0000) | 1/1 (1.0000) | 3.3856 |
| `metric_rubric` | test | Motivation, Guidance & Positive Feedback | 4 | 2 (0.5000) | 2/4 (0.5000) | 0.3687 |
| `metric_rubric` | test | Higher-Order Thinking & Skill Development | 6 | 2 (0.3333) | 2/6 (0.3333) | -0.2724 |
| `metric_rubric` | test | Reasoning Process Rigor | 10 | 2 (0.2000) | 1/1 (1.0000) | -5.9209 |
| `metric_rubric` | test | Personalization, Adaptation & Learning Support | 2 | 0 (0.0000) | 0/2 (0.0000) | -0.1141 |
| `qmr` | dev | Instruction Following & Task Completion | 7 | 7 (1.0000) | 7/7 (1.0000) | 3.0907 |
| `qmr` | dev | Error Identification & Correction Precision | 5 | 5 (1.0000) | 4/4 (1.0000) | 2.8881 |
| `qmr` | dev | Clarity, Simplicity & Inspiration | 4 | 4 (1.0000) | 4/4 (1.0000) | 1.8851 |
| `qmr` | dev | Scenario Element Integration | 4 | 4 (1.0000) | 4/4 (1.0000) | 1.3816 |
| `qmr` | dev | Basic Factual Accuracy | 2 | 2 (1.0000) | 2/2 (1.0000) | 4.8518 |
| `qmr` | dev | Content Relevance & Scope Control | 2 | 2 (1.0000) | 2/2 (1.0000) | 4.9602 |
| `qmr` | dev | Motivation, Guidance & Positive Feedback | 9 | 2 (0.2222) | 1/4 (0.2500) | -0.5318 |
| `qmr` | dev | Reasoning Process Rigor | 16 | 3 (0.1875) | 3/3 (1.0000) | -1.5253 |
| `qmr` | dev | Higher-Order Thinking & Skill Development | 7 | 0 (0.0000) | 0/7 (0.0000) | -1.4552 |
| `qmr` | dev | Personalization, Adaptation & Learning Support | 1 | 0 (0.0000) | 0/1 (0.0000) | -0.1639 |
| `qmr` | test | Instruction Following & Task Completion | 5 | 5 (1.0000) | 5/5 (1.0000) | 3.0749 |
| `qmr` | test | Motivation, Guidance & Positive Feedback | 4 | 4 (1.0000) | 4/4 (1.0000) | 0.7738 |

## Boundary-Key Stability

WARNING: `229` boundary_key groups have non-negligible tau variation (`max_abs_tau_diff > 1e-5`).
This is not a monotonicity violation; it means the same saved boundary context can still produce
slightly different tau values across prediction rows.

## qmr vs metric_rubric

- `qmr` includes question text in the boundary tower; `metric_rubric` only uses metric and rubric.
- Compare dev/test low-to-high and label-2 prediction distributions above to decide whether question-specific boundaries improve stability.

## Concise Conclusion

Exp16A enforces ordered thresholds with zero monotonic violation, but the same boundary_key can
still produce non-identical tau values in saved predictions. Treat this boundary-key warning as part
of the RQ1 diagnosis, not as a solved stability result. Label-2 recall and low-to-high failures
remain the key weakness. The failure diagnosis should be read as evidence for whether the next step
should calibrate tau thresholds or improve the quality score separation before moving to risk-aware
RQ2 methods.
