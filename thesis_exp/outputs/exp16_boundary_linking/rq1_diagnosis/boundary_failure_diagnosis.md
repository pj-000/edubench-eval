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

Interpretation rule:

- `< 1e-4`: treat as numerical noise.
- `> 1e-3`: warning; batch/order/numerical effects are no longer negligible.
- `> 1e-2`: serious warning; inspect cache/export implementation and boundary reuse.

WARNING: `229` boundary_key groups have non-negligible tau variation (`max_abs_tau_diff > 1e-5`).
This is not a monotonicity violation; it means the same saved boundary context can still produce
slightly different tau values across prediction rows.
SERIOUS WARNING: `89` groups have `max_abs_tau_diff > 1e-2`.

### Boundary-Key Stability Quantile Summary

| variant | split | groups | unstable | warning | serious | median max diff | p95 max diff | max diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `metric_rubric` | dev | 24 | 16 | 16 | 7 | 0.004477 | 0.017705 | 0.026184 |
| `metric_rubric` | test | 24 | 21 | 21 | 8 | 0.006787 | 0.023061 | 0.026184 |
| `qmr` | dev | 224 | 99 | 96 | 42 | 0.000000 | 0.031530 | 0.058834 |
| `qmr` | test | 222 | 93 | 93 | 32 | 0.000000 | 0.018538 | 0.051087 |

### Top-20 Most Unstable Boundary-Key Groups

| variant | split | n | max diff | tau | mean diff | p95 diff | boundary_key | examples |
|---|---|---:|---:|---|---:|---:|---|---|
| `qmr` | dev | 5 | 0.058834 | tau4 | 0.022316 | 0.058834 | `2ee62978bf08` | 143f844f0de67a82fe8b1355e71144c090c2a8b4, 9daae5db6dddbbda0ed534c93d6e903453c4ebe4, c066a767cda6a5b4af048ba3e374c8295bdd922d |
| `qmr` | dev | 5 | 0.056682 | tau4 | 0.021798 | 0.056682 | `34f2a2f349a2` | 29743b55549957b2840fe53b5f703ce487b5c1ce, 443ba278d4c2d3652bff93c57e0dece256cce4d7, 5faf89cb0198762be0a7f4c56e46b827ab74def6 |
| `qmr` | test | 5 | 0.051087 | tau4 | 0.015871 | 0.051087 | `d462b83bcf11` | 1759a711276fe47fb06f64cfcd905996d0665729, 2b5607dae4cb4c05da55b8706f3483149295b17e, 90b7d1926e47fc40035e61569b3c6c4c7287d54c |
| `qmr` | dev | 5 | 0.049035 | tau4 | 0.016514 | 0.039135 | `62972a567e5f` | 09691f5315938a4adbf7895ed89748b85f462522, 387925329b013cabf37b462cae965a905dbccbe7, 9d751f7afec36098b7bef133117cd1747a954e18 |
| `qmr` | dev | 5 | 0.043637 | tau3 | 0.019907 | 0.043637 | `92c2cb633e24` | 15a5c95f8d439088d6b196caf73638e76f0cdfda, 438c594ed058cc5c0705a71f4ed30835391a85a1, 7a864fd83c7c351a4a8ba76d82ffe42fde099021 |
| `qmr` | dev | 5 | 0.042760 | tau3 | 0.015193 | 0.042760 | `1ddc60fb33f9` | 386128679d1defd89ca1103a9138eea17e2c7ad3, 5018e8582681cfc20328cedc0fc2b6aa2d6ea2c3, 58a5c8a775f59e853173fa90373bf683a3caec38 |
| `qmr` | dev | 5 | 0.039721 | tau3 | 0.021706 | 0.039721 | `f4439a271cd8` | 0acfae215fbf65b18d9efae29cc65c39314c9fa7, 1e6be4d3a41606c04cd87c96e507d4e53e8f04e1, 2c52964714627fa961cc72666ebccf0da2ac765b |
| `qmr` | dev | 5 | 0.038505 | tau2 | 0.012770 | 0.036895 | `912555bd2b4a` | 057fb10dadb0ef587ef63e8c0692fb069e747f33, 7f474f93d1bb8c2770cc7c8ab8464c1d7389c3cb, abd894527f401792d7a5cb4438ee15420de3ea1d |
| `qmr` | dev | 5 | 0.035709 | tau3 | 0.018938 | 0.035709 | `6df27a99fbc2` | 053de98886f60ab5ec51e900d2ec93bc42fee4d5, 753287728aff9790683dbd25f511d837e41e54b6, 8bed9df40b0527c32f815e91c2df8cb414309911 |
| `qmr` | dev | 5 | 0.035012 | tau2 | 0.013198 | 0.035012 | `9c0c0a7161c6` | 09f0a5e6798734f1859446596f6223faaa5d8495, 0e9def4fcbc41318c97e0055e6413b82dbe0a145, 8e78ca2c20375dd8efcf91036ac7efd7ca08d9b1 |
| `qmr` | dev | 5 | 0.033892 | tau3 | 0.019552 | 0.033892 | `022e83b668d7` | 0361cc9784ec44f7e093970b9f81493a6df9adbc, 35e1d8543c7ce6e25676e1a90d134cf2212ab356, 50a23e9c860fdd45abf3315f64af742c6032bef8 |
| `qmr` | dev | 5 | 0.033768 | tau2 | 0.011752 | 0.033768 | `38fc4f44f33d` | 130b9044b1debd72bd2bb3dabbcfc7081a1966bc, 1a015be39daf65db2c1ff112b72c2b04f851ac48, 4ebfdf536e6f81585029508942e48f3ede5bde12 |
| `qmr` | test | 5 | 0.032300 | tau4 | 0.011803 | 0.032300 | `7cef03ed033b` | 323a91df656b63dc795e34769c0695bb4cf30656, 4120359499d292030549af56f9241bdf529b689b, 5546f1ea2b064f7348025fe8a8b6478851600ca6 |
| `qmr` | dev | 5 | 0.031878 | tau3 | 0.009809 | 0.018460 | `518449ae5b6b` | 3191ea4612e1f06e74580be14689c4e60b3aa5ec, a0a401e9defc721901dffdbb9d9bcf0acaefb9a2, cd15da1795d9a9c658f6c9a7fddd059483bfbd8c |
| `qmr` | test | 5 | 0.031425 | tau4 | 0.011527 | 0.031425 | `8ab6d5fbaf55` | 12747760c9e36c869507f117c7053340bdfad0cd, 238c21ac5f03fe2ff15e95d3e200c645782b3190, 43be896a6e5e7051694a81a781727c144ca0b282 |
| `qmr` | test | 5 | 0.031250 | tau1 | 0.016356 | 0.031250 | `237ff5e6d432` | 674c0c6bbf476257e33649e7d303ac1697e09607, 8944f4a89dd8cf4e50ef83d63945e04592b52d80, adc3b10884081f459539cf757fa8f5710ca3ce75 |
| `qmr` | test | 5 | 0.031250 | tau1 | 0.010807 | 0.031250 | `d4f7719927ce` | 4bc5d554ec0ee6b28528fe98b9c21a6bc10853bd, 7f52dd1d97f054bbb35033cf92a6e23ba1f65fdc, 8ebade06973f5ef215a59951c5feb5f063657b22 |
| `qmr` | dev | 5 | 0.029556 | tau4 | 0.004939 | 0.029556 | `d5ff9fc8ea92` | 0b852b93df7d7ad0db7ec99d276d3e4c2e6d53c9, 16bea1c1cd9d748355b4747cb6ce241d7fcfbc2c, 590c7f542d0b4c8788f36aa8afe67719d4d593a3 |
| `metric_rubric` | dev | 16 | 0.026184 | tau4 | 0.004428 | 0.026184 | `6049ce916c27` | 08878a554232c03876dcd910306e1aead6bc04c4, 17b2ac651081219e46d26ed2caee675d63e8e54b, 1a67eecdaf74f2088dc0d0122ffbd507d87e6ec4 |
| `metric_rubric` | test | 43 | 0.026184 | tau4 | 0.007469 | 0.024584 | `6049ce916c27` | 009a70f14432d291d4ccb15a5281c1e8b7e4baf5, 025440cebee6f5ce51c1eb618e7ec69eaa724e5e, 094d533fd29ce8e42fe29bc8c70a86a007ddc194 |

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
