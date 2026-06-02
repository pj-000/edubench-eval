# Exp1: Evaluator-vs-Human Audit

## 1. Purpose

This experiment audits agreement between existing automatic judge predictions and human scores on
the locked Exp0.1 paper-like test split. It reproduces the PDF-style low-score blind spot analysis
and extends it with metric, scenario, subject, language, education-level, and generator-model
strata. It is intended as the Chapter 4 target analysis for downstream training and calibration
experiments.

## 2. Inputs

| item | value |
| --- | --- |
| main dataset | `thesis_exp/data/processed/edubench_scoring_all.jsonl` |
| dataset name | `edubench_audit_human_scored_subset` |
| main split | `thesis_exp/data/splits/paper_like_triple_seed42/test.jsonl` |
| test rows | 2218 |
| human reference | `human_mean_5` and rounded `label_5` |
| Exp0.1 references | `thesis_exp/outputs/exp00_data/review_package.md`, `thesis_exp/outputs/exp00_data/data_card.md`, `thesis_exp/outputs/exp00_data/leakage_report.md`, `thesis_exp/outputs/exp00_data/sanity_check_exp00_reference.md` |

## 3. Judge Score Inventory and Alignment

Evaluators found: EduBenchEvaluator, GPT-4o, DeepSeek-R1, DeepSeek-V3, QwQ-plus.
Evaluators missing: None.

| evaluator | n_test | n_aligned | coverage | n_valid_score | valid_score_rate | n_missing | n_invalid | primary_alignment_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EduBenchEvaluator | 2218 | 2218 | 1.0 | 2218 | 1.0 | 0 | 0 | record_id |
| GPT-4o | 2218 | 2218 | 1.0 | 2192 | 0.9882777276825968 | 0 | 26 | record_id |
| DeepSeek-R1 | 2218 | 2218 | 1.0 | 2218 | 1.0 | 0 | 0 | record_id |
| DeepSeek-V3 | 2218 | 2218 | 1.0 | 2218 | 1.0 | 0 | 0 | record_id |
| QwQ-plus | 2218 | 2218 | 1.0 | 2196 | 0.9900811541929666 | 0 | 22 | record_id |

Missing evaluator warnings:

- None.

No missing or invalid prediction was filled. Synthetic/sample files are inventoried only as excluded
sources.

## 4. Overall Agreement with Human Scores

| evaluator | n_valid | MAE | RMSE | Signed Bias | Exact Match | Within-1 Accuracy | Macro-F1 | Weighted-F1 | Quadratic Weighted Kappa | Kendall tau | Spearman rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EduBenchEvaluator | 2218 | 0.4358280733393447 | 0.7665614595682864 | 0.25278028253682 | 0.7240757439134355 | 0.9391343552750224 | 0.5483548352955511 | 0.704765248584029 | 0.532925403514833 | 0.5051636452404249 | 0.5613714568069045 |
| GPT-4o | 2192 | 0.6027980535279807 | 0.9722424977610036 | 0.4802311435523114 | 0.5752737226277372 | 0.9128649635036497 | 0.229723455328664 | 0.5077540388286318 | 0.1800161173407197 | 0.2767656507351872 | 0.3124140824073464 |
| DeepSeek-R1 | 2218 | 0.5952810339645327 | 0.9698645321385512 | 0.3379921851517884 | 0.584761045987376 | 0.9125338142470696 | 0.3229285972345866 | 0.545181600588471 | 0.2765074920629188 | 0.3154341848109602 | 0.3588622700941416 |
| DeepSeek-V3 | 2218 | 0.5807033363390441 | 0.9535342323752156 | 0.4601743312293357 | 0.6032461677186655 | 0.9156898106402164 | 0.2736265299451729 | 0.5412365867518423 | 0.22654875290799 | 0.3243146328338703 | 0.3652136561871008 |
| QwQ-plus | 2196 | 0.5980570734669096 | 0.98977774669706 | 0.4074074074074075 | 0.604735883424408 | 0.9075591985428052 | 0.3070212471710575 | 0.551692410643831 | 0.2282801098661906 | 0.2992004212761405 | 0.3384829832477977 |

Lowest MAE: EduBenchEvaluator (0.436).
Highest Exact Match: EduBenchEvaluator (0.724).
Largest absolute signed bias: GPT-4o (0.480).

## 5. Low-score Blind Spot

| evaluator | n_valid_low | low_exact_match | low_recall | low_MAE | low_signed_bias | low_overestimation_rate | low_severe_overestimation_rate | low_to_high_rate | mean_pred_low | mean_human_low |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EduBenchEvaluator | 103 | 0.3495145631067961 | 0.3883495145631068 | 1.8511326860841424 | 1.812297734627832 | 0.6407766990291263 | 0.5339805825242718 | 0.5339805825242718 | 3.320388349514563 | 1.5080906148867312 |
| GPT-4o | 103 | 0.0 | 0.0 | 3.0550161812297736 | 3.0550161812297736 | 1.0 | 0.9223300970873788 | 0.912621359223301 | 4.563106796116505 | 1.5080906148867312 |
| DeepSeek-R1 | 103 | 0.0970873786407767 | 0.0970873786407767 | 2.799352750809062 | 2.7734627831715213 | 0.9029126213592232 | 0.7961165048543689 | 0.7669902912621359 | 4.281553398058253 | 1.5080906148867312 |
| DeepSeek-V3 | 103 | 0.0485436893203883 | 0.0485436893203883 | 2.9385113268608416 | 2.9385113268608416 | 0.9514563106796116 | 0.912621359223301 | 0.912621359223301 | 4.446601941747573 | 1.5080906148867312 |
| QwQ-plus | 101 | 0.0693069306930693 | 0.0792079207920792 | 3.06930693069307 | 3.016501650165017 | 0.9207920792079208 | 0.8613861386138614 | 0.8514851485148515 | 4.524752475247524 | 1.508250825082508 |

Per-bin Acc@1/Acc@2:

| evaluator | label_5 | n_valid | accuracy | mean_pred | signed_bias |
| --- | --- | --- | --- | --- | --- |
| EduBenchEvaluator | 1 | 56 | 0.4464285714285714 | 2.9285714285714284 | 1.9285714285714288 |
| EduBenchEvaluator | 2 | 47 | 0.2340425531914893 | 3.7872340425531914 | 1.6737588652482274 |
| GPT-4o | 1 | 56 | 0.0 | 4.857142857142857 | 3.857142857142857 |
| GPT-4o | 2 | 47 | 0.0 | 4.212765957446808 | 2.0992907801418443 |
| DeepSeek-R1 | 1 | 56 | 0.0892857142857142 | 4.535714285714286 | 3.5357142857142856 |
| DeepSeek-R1 | 2 | 47 | 0.1063829787234042 | 3.978723404255319 | 1.865248226950355 |
| DeepSeek-V3 | 1 | 56 | 0.0892857142857142 | 4.589285714285714 | 3.5892857142857144 |
| DeepSeek-V3 | 2 | 47 | 0.0 | 4.276595744680851 | 2.163120567375887 |
| QwQ-plus | 1 | 55 | 0.0 | 4.963636363636364 | 3.963636363636364 |
| QwQ-plus | 2 | 46 | 0.1521739130434782 | 4.0 | 1.8840579710144931 |

The central failure mode remains low-score overestimation: true labels 1 and 2 are much harder than
high-score items, and several evaluators push low-scored answers into mid/high predicted labels.

## 6. Calibration Failure

Calibration is summarized by mean predicted score per true label and signed bias per true label.
Positive signed bias, especially for labels 1 and 2, indicates systematic overestimation. High-score
items are generally preserved more reliably than low-score items.

## 7. Metric-level Differences

Highest mean MAE metrics:

| metric_canonical | MAE |
| --- | --- |
| Reasoning Process Rigor | 1.3936910525941186 |
| Motivation, Guidance & Positive Feedback | 0.7251717108083752 |
| Higher-Order Thinking & Skill Development | 0.7183156424273985 |
| Scenario Element Integration | 0.6741058655221746 |
| Personalization, Adaptation & Learning Support | 0.5535487758945388 |

Kendall tau may be undefined for strata where predictions or human scores are constant; those cells
are reported as NaN rather than failing the pipeline.

## 8. Scenario-level Differences

Highest mean MAE scenarios:

| scenario_canonical | MAE |
| --- | --- |
| Question Answering | 1.064247311827957 |
| Personalized Learning Support | 0.5877563333103359 |
| Error Correction | 0.5340105383239326 |
| Personalized Content Creation | 0.5037735849056604 |
| Idea Provision | 0.5014825131001661 |

## 9. Subject-level Differences

Highest mean MAE subjects:

| subject_canonical | MAE |
| --- | --- |
| Business Administration | 0.9354382355381782 |
| Clinical Medicine | 0.8115384615384617 |
| Military Science | 0.7083333333333334 |
| History | 0.670306137466024 |
| Applied Economics | 0.6424242424242423 |
| Computer Science | 0.6335424836601306 |
| Geography | 0.6312827523353841 |
| Law | 0.6044324324324324 |

Warning: subject metadata comes from local enriched metadata and should be treated as stratified
audit metadata.

## 10. Comparison with PDF Reference

PDF trend reproduced: **YES**.

| evaluator | metric_name | pdf_reference | current_value | delta | status |
| --- | --- | --- | --- | --- | --- |
| EduBenchEvaluator | MAE | 0.43 | 0.4358280733393447 | 0.0058280733393447 | matched_trend |
| EduBenchEvaluator | Signed Bias | 0.246 | 0.25278028253682 | 0.00678028253682 | matched_trend |
| EduBenchEvaluator | Exact Match | 0.725 | 0.7240757439134355 | -0.0009242560865644 | matched_trend |
| EduBenchEvaluator | Kendall tau | 0.508 | 0.5051636452404249 | -0.0028363547595751 | matched_trend |
| EduBenchEvaluator | Bin Agreement | 0.897 | NaN | NaN | not_comparable |
| DeepSeek-V3 | MAE | 0.576 | 0.5807033363390441 | 0.0047033363390441 | matched_trend |
| DeepSeek-V3 | Signed Bias | 0.458 | 0.4601743312293357 | 0.0021743312293357 | matched_trend |
| DeepSeek-V3 | Exact Match | 0.602 | 0.6032461677186655 | 0.0012461677186654 | matched_trend |
| DeepSeek-V3 | Kendall tau | 0.326 | 0.3243146328338703 | -0.0016853671661297 | matched_trend |
| DeepSeek-V3 | Bin Agreement | 0.867 | NaN | NaN | not_comparable |
| DeepSeek-R1 | MAE | 0.589 | 0.5952810339645327 | 0.0062810339645327 | matched_trend |
| DeepSeek-R1 | Signed Bias | 0.335 | 0.3379921851517884 | 0.0029921851517883 | matched_trend |
| DeepSeek-R1 | Exact Match | 0.585 | 0.584761045987376 | -0.0002389540126239 | matched_trend |
| DeepSeek-R1 | Kendall tau | 0.319 | 0.3154341848109602 | -0.0035658151890398 | matched_trend |
| DeepSeek-R1 | Bin Agreement | 0.854 | NaN | NaN | not_comparable |
| QwQ-plus | MAE | 0.593 | 0.5980570734669096 | 0.0050570734669096 | matched_trend |
| QwQ-plus | Signed Bias | 0.402 | 0.4074074074074075 | 0.0054074074074074 | matched_trend |
| QwQ-plus | Exact Match | 0.604 | 0.604735883424408 | 0.000735883424408 | matched_trend |
| QwQ-plus | Kendall tau | 0.301 | 0.2992004212761405 | -0.0017995787238594 | matched_trend |
| QwQ-plus | Bin Agreement | 0.86 | NaN | NaN | not_comparable |
| GPT-4o | MAE | 0.598 | 0.6027980535279807 | 0.0047980535279806 | matched_trend |
| GPT-4o | Signed Bias | 0.475 | 0.4802311435523114 | 0.0052311435523114 | matched_trend |
| GPT-4o | Exact Match | 0.575 | 0.5752737226277372 | 0.0002737226277372 | matched_trend |
| GPT-4o | Kendall tau | 0.278 | 0.2767656507351872 | -0.0012343492648128 | matched_trend |
| GPT-4o | Bin Agreement | 0.868 | NaN | NaN | not_comparable |
| EduBenchEvaluator | Acc@1 | 48.1 | 44.642857142857146 | -3.4571428571428555 | matched_trend |
| EduBenchEvaluator | Acc@2 | 23.4 | 23.404255319148938 | 0.0042553191489389 | matched_trend |
| EduBenchEvaluator | Acc@3 | 21.1 | 21.1340206185567 | 0.0340206185566991 | matched_trend |
| EduBenchEvaluator | Acc@4 | 66.1 | 65.97222222222221 | -0.12777777777778 | matched_trend |
| EduBenchEvaluator | Acc@5 | 87.7 | 87.76019983347211 | 0.0601998334721116 | matched_trend |
| QwQ-plus | Acc@1 | 0.0 | 0.0 | 0.0 | matched_trend |
| QwQ-plus | Acc@2 | 15.2 | 15.217391304347828 | 0.0173913043478286 | matched_trend |
| QwQ-plus | Acc@3 | 16.3 | 16.315789473684212 | 0.0157894736842116 | matched_trend |
| QwQ-plus | Acc@4 | 28.2 | 28.28854314002829 | 0.088543140028289 | matched_trend |
| QwQ-plus | Acc@5 | 91.0 | 90.98497495826378 | -0.0150250417362229 | matched_trend |
| DeepSeek-V3 | Acc@1 | 7.7 | 8.928571428571429 | 1.2285714285714286 | matched_trend |
| DeepSeek-V3 | Acc@2 | 0.0 | 0.0 | 0.0 | matched_trend |
| DeepSeek-V3 | Acc@3 | 4.1 | 4.123711340206185 | 0.0237113402061854 | matched_trend |
| DeepSeek-V3 | Acc@4 | 31.4 | 31.52777777777778 | 0.12777777777778 | matched_trend |
| DeepSeek-V3 | Acc@5 | 91.4 | 91.42381348875936 | 0.0238134887593588 | matched_trend |
| DeepSeek-R1 | Acc@1 | 7.7 | 8.928571428571429 | 1.2285714285714286 | matched_trend |
| DeepSeek-R1 | Acc@2 | 10.6 | 10.638297872340424 | 0.0382978723404257 | matched_trend |
| DeepSeek-R1 | Acc@3 | 18.0 | 18.04123711340206 | 0.0412371134020617 | matched_trend |
| DeepSeek-R1 | Acc@4 | 30.6 | 30.416666666666664 | -0.1833333333333371 | matched_trend |
| DeepSeek-R1 | Acc@5 | 86.0 | 86.01165695253955 | 0.0116569525395533 | matched_trend |
| GPT-4o | Acc@1 | 0.0 | 0.0 | 0.0 | matched_trend |
| GPT-4o | Acc@2 | 0.0 | 0.0 | 0.0 | matched_trend |
| GPT-4o | Acc@3 | 5.9 | 5.851063829787234 | -0.0489361702127659 | matched_trend |
| GPT-4o | Acc@4 | 25.1 | 25.17580872011252 | 0.0758087201125192 | matched_trend |
| GPT-4o | Acc@5 | 90.0 | 90.0 | 0.0 | matched_trend |

The comparison checks whether the main trend is reproduced rather than forcing exact numerical
matches. Differences can come from repaired/reconstructed split details, alignment source choice,
and unavailable PDF bin-agreement definitions.

## 11. Implications for Exp2-Exp7

- Exp2 should reproduce or establish the EduBenchEvaluator CE baseline.
- Exp3 should test rubric-aware inputs.
- Exp5 should test low-score-sensitive loss.
- Exp6 should use synthetic low-score augmentation only as a controlled follow-up, not as Exp1 data.
- Exp7 should address calibration.
- Later experiments should not rely only on overall accuracy.

## 12. Figures and Tables

Core figures:

- `thesis_exp/outputs/exp01_audit/figures/fig01_bias_by_true_score.png`
- `thesis_exp/outputs/exp01_audit/figures/fig01_calibration_curve.png`
- `thesis_exp/outputs/exp01_audit/figures/fig01_confusion_matrix_DeepSeekR1.png`
- `thesis_exp/outputs/exp01_audit/figures/fig01_confusion_matrix_DeepSeekV3.png`
- `thesis_exp/outputs/exp01_audit/figures/fig01_confusion_matrix_EduBenchEvaluator.png`
- `thesis_exp/outputs/exp01_audit/figures/fig01_confusion_matrix_GPT4o.png`
- `thesis_exp/outputs/exp01_audit/figures/fig01_confusion_matrix_QwQPlus.png`
- `thesis_exp/outputs/exp01_audit/figures/fig01_evaluator_exact_kendall.png`
- `thesis_exp/outputs/exp01_audit/figures/fig01_evaluator_overall_mae_bias.png`
- `thesis_exp/outputs/exp01_audit/figures/fig01_low_score_overestimation_rate.png`
- `thesis_exp/outputs/exp01_audit/figures/fig01_mean_pred_by_true_label.png`
- `thesis_exp/outputs/exp01_audit/figures/fig01_metric_mae_heatmap.png`
- `thesis_exp/outputs/exp01_audit/figures/fig01_per_bin_accuracy.png`
- `thesis_exp/outputs/exp01_audit/figures/fig01_scenario_mae_heatmap.png`
- `thesis_exp/outputs/exp01_audit/figures/fig01_subject_mae_heatmap.png`
- `thesis_exp/outputs/exp01_audit/figures/fig01_test_label_distribution.png`

Tables:

- `thesis_exp/outputs/exp01_audit/tables/alignment_coverage.csv`
- `thesis_exp/outputs/exp01_audit/tables/education_level_metrics.csv`
- `thesis_exp/outputs/exp01_audit/tables/evaluator_metrics.csv`
- `thesis_exp/outputs/exp01_audit/tables/generator_model_level_metrics.csv`
- `thesis_exp/outputs/exp01_audit/tables/high_score_metrics.csv`
- `thesis_exp/outputs/exp01_audit/tables/invalid_judge_outputs.csv`
- `thesis_exp/outputs/exp01_audit/tables/judge_score_inventory.csv`
- `thesis_exp/outputs/exp01_audit/tables/language_level_metrics.csv`
- `thesis_exp/outputs/exp01_audit/tables/low_score_metrics.csv`
- `thesis_exp/outputs/exp01_audit/tables/metric_level_metrics.csv`
- `thesis_exp/outputs/exp01_audit/tables/missing_judge_scores.csv`
- `thesis_exp/outputs/exp01_audit/tables/pdf_reference_comparison.csv`
- `thesis_exp/outputs/exp01_audit/tables/per_bin_metrics.csv`
- `thesis_exp/outputs/exp01_audit/tables/preflight_check_exp01.csv`
- `thesis_exp/outputs/exp01_audit/tables/scenario_level_metrics.csv`
- `thesis_exp/outputs/exp01_audit/tables/subject_level_metrics.csv`

## 13. Limitations

- No new model is trained.
- No API is called.
- Only existing judge predictions are used.
- Missing evaluator predictions must be supplied in later experiments rather than inferred here.
- The paper-like split has the Exp0.1 question-overlap warning context.
- Subject-level provenance depends on local enriched metadata.
