# Exp3: Rubric-aware Input Ablation

## 1. Purpose
Exp3 is an input ablation experiment. It tests whether adding question, metric, rubric, and
metadata fields improves agreement between a local education scoring model and human scores.

## 2. Relation to Exp2
Exp2 is the Q+A+metric Cross Entropy baseline. Exp3 keeps the same 5-class CE objective,
model family, split, and checkpoint selection. A2 is the Exp2-compatible template and is
reused by default instead of retrained.

## 3. Input Templates
| ablation_id | template_name | input_fields | default |
| --- | --- | --- | --- |
| A0 | A0_answer_only | answer | disabled_initial |
| A1 | A1_question_answer | question, answer | disabled_initial |
| A2 | A2_question_answer_metric | question, answer, metric_canonical | reuse_exp02 |
| A3 | A3_question_answer_metric_rubric | question, answer, metric_canonical, rubric_text | core_formal |
| A4 | A4_question_answer_metric_rubric_metadata | scenario_canonical, subject_canonical, education_level_canonical, lan... | core_formal |

A4 intentionally excludes generator_model, answer_model, human labels, and chain-of-thought.

## 4. Rubric Source and Quality Audit
Rubric coverage: 5536/5536.
Rubric mode: **corrected**.
Rubric quality status: **PASS**.
Special zh SEI vs IFTC check: **PASS**.
Human confirmation needed: **NO**.
Raw rubric text is read from the split row field. The audit shows it is constant within
each metric/language group, so Exp3 treats it as metric-level rubric description, not
sample-specific human annotation. The active rubric mode may override known defective
metric/language rows before A3/A4 prompts are built.

Source audit: `thesis_exp/outputs/exp03_input_ablation/reports/rubric_source_audit.md`
Quality audit: `thesis_exp/outputs/exp03_input_ablation/reports/rubric_quality_audit.md`
Repair trace: `thesis_exp/outputs/exp03_input_ablation/reports/rubric_repair_source_trace.md`

## 5. Dataset and Training Setup
The locked Exp0.1 paper-like triple split is used: train=2654, dev=664, test=2218. Labels
remain label_5 in {1,2,3,4,5}, mapped to class indices 0..4 for CE training. Test is used
only after selecting the best checkpoint on dev Exact Match.

## 6. Smoke Test Results
Smoke test status: **PASS**.
Server smoke can start: **YES.**

## 7. Available Ablation Results
| ablation_id | template_name | status | test_accuracy | test_MAE_label | test_kendall_tau | test_low_to_high_rate | mean_token_length | truncation_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | A0_answer_only | pending |  |  |  |  | 280.93081647398844 | 0.0 |
| A1 | A1_question_answer | pending |  |  |  |  | 355.8553106936416 | 0.0 |
| A2 | A2_question_answer_metric | reused_exp02 | 0.7299368800721371 | 0.4238052299368801 | 0.5692855146538734 | 0.5339805825242718 | 370.06123554913296 | 0.0 |
| A3 | A3_question_answer_metric_rubric | completed | 0.7168620378719567 | 0.4236549443943492 | 0.5914073850965026 | 0.39805825242718446 | 477.8860187861272 | 0.0 |
| A4 | A4_question_answer_metric_rubric_metadata | completed | 0.7412082957619477 | 0.4030658250676285 | 0.5940582341762678 | 0.44660194174757284 | 502.32947976878614 | 0.0 |

## 8. Low-score Analysis
| ablation_id | template_name | status | test_acc_at_1 | test_acc_at_2 | test_low_to_high_rate |
| --- | --- | --- | --- | --- | --- |
| A0 | A0_answer_only | pending |  |  |  |
| A1 | A1_question_answer | pending |  |  |  |
| A2 | A2_question_answer_metric | reused_exp02 | 0.17857142857142858 | 0.2553191489361702 | 0.5339805825242718 |
| A3 | A3_question_answer_metric_rubric | completed | 0.08928571428571429 | 0.3617021276595745 | 0.39805825242718446 |
| A4 | A4_question_answer_metric_rubric_metadata | completed | 0.3392857142857143 | 0.3829787234042553 | 0.44660194174757284 |

The key Exp3 low-score metric is low_to_high_rate: the fraction of true label 1/2 samples
predicted as 4/5. A3/A4 should be judged partly by whether rubric or metadata reduces this
overestimation without collapsing high-score accuracy.

## 9. Metric-level Analysis
Metric-level delta tables are written after available runs are collected. The A3 - A2 table
is the main place to inspect which dimensions benefit from rubric-aware input.

## 10. Token Length and Truncation
Token truncation warning: none.
Output formatting status: **PASS**.
Token lengths are estimated when no tokenizer is available and recomputed with the model
tokenizer when supplied during dataset building or training.

## 11. Implications for Exp4-Exp7
If A3 improves low-score recognition without material truncation, Exp4-Exp7 should use A3
as the default input. If A4 helps only in distribution-specific slices, metadata should be
reported as useful but potentially distribution-dependent.

## 12. Limitations
- Exp3 changes only inputs, not loss or model architecture.
- Metadata may introduce distribution dependence.
- Subject metadata provenance is derived from Exp0 alignment and should be described cautiously.
- Question-split robustness remains a later validation target.
