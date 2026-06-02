# Exp1 Review Package

| item | value |
| --- | --- |
| Can Exp2 start? | YES |
| Test set rows | 2218 |
| Evaluators found | EduBenchEvaluator, GPT-4o, DeepSeek-R1, DeepSeek-V3, QwQ-plus |
| Evaluators missing | None |
| PDF trend reproduced? | YES |

## Alignment Coverage

| evaluator | n_test | n_aligned | coverage | n_valid_score | valid_score_rate | n_missing | n_invalid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EduBenchEvaluator | 2218 | 2218 | 1.0 | 2218 | 1.0 | 0 | 0 |
| GPT-4o | 2218 | 2218 | 1.0 | 2192 | 0.9882777276825968 | 0 | 26 |
| DeepSeek-R1 | 2218 | 2218 | 1.0 | 2218 | 1.0 | 0 | 0 |
| DeepSeek-V3 | 2218 | 2218 | 1.0 | 2218 | 1.0 | 0 | 0 |
| QwQ-plus | 2218 | 2218 | 1.0 | 2196 | 0.9900811541929666 | 0 | 22 |

## Main Evaluator Metrics

| evaluator | n_valid | MAE | Signed Bias | Exact Match | Within-1 Accuracy | Kendall tau | Spearman rho |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EduBenchEvaluator | 2218 | 0.4358280733393447 | 0.25278028253682 | 0.7240757439134355 | 0.9391343552750224 | 0.5051636452404249 | 0.5613714568069045 |
| GPT-4o | 2192 | 0.6027980535279807 | 0.4802311435523114 | 0.5752737226277372 | 0.9128649635036497 | 0.2767656507351872 | 0.3124140824073464 |
| DeepSeek-R1 | 2218 | 0.5952810339645327 | 0.3379921851517884 | 0.584761045987376 | 0.9125338142470696 | 0.3154341848109602 | 0.3588622700941416 |
| DeepSeek-V3 | 2218 | 0.5807033363390441 | 0.4601743312293357 | 0.6032461677186655 | 0.9156898106402164 | 0.3243146328338703 | 0.3652136561871008 |
| QwQ-plus | 2196 | 0.5980570734669096 | 0.4074074074074075 | 0.604735883424408 | 0.9075591985428052 | 0.2992004212761405 | 0.3384829832477977 |

## Low-score Blind Spot

| evaluator | n_valid_low | low_exact_match | low_recall | low_signed_bias | low_to_high_rate | mean_pred_low |
| --- | --- | --- | --- | --- | --- | --- |
| EduBenchEvaluator | 103 | 0.3495145631067961 | 0.3883495145631068 | 1.812297734627832 | 0.5339805825242718 | 3.320388349514563 |
| GPT-4o | 103 | 0.0 | 0.0 | 3.0550161812297736 | 0.912621359223301 | 4.563106796116505 |
| DeepSeek-R1 | 103 | 0.0970873786407767 | 0.0970873786407767 | 2.7734627831715213 | 0.7669902912621359 | 4.281553398058253 |
| DeepSeek-V3 | 103 | 0.0485436893203883 | 0.0485436893203883 | 2.9385113268608416 | 0.912621359223301 | 4.446601941747573 |
| QwQ-plus | 101 | 0.0693069306930693 | 0.0792079207920792 | 3.016501650165017 | 0.8514851485148515 | 4.524752475247524 |

## Blockers Before Exp2

- None blocking Exp2 baseline setup.

## Recommended Next Step

Start Exp2 by locking the CE baseline protocol and preserving this Exp1 low-score audit as the
primary evaluator-bias target.
