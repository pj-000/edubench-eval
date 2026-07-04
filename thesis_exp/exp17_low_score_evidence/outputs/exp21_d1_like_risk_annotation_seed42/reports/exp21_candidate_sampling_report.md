# Exp21 D1-like Risk Annotation Candidate Sampling

Exp21 constructs annotation candidates only. It does not train, does not read test,
and does not use human rationale as model input.

## Exp20C Failure Modes Observed

- Automatic downgrade reduced aggregate low-to-high but demoted many true high-score cases.
- D1 hidden residual high-score predictions remain high, so the gate lacks enough precision/recall.
- The candidate package therefore includes both likely high-risk low cases and high-score false positives.

## Dev Audit Package

- downgraded_gold_high_to_3: 35
- rescued_d1_low_to_high: 5
- rescued_low_to_high: 1
- residual_d1_pred_ge4: 16
- residual_low_to_high: 1
- unchanged_safe_high_control: 40

## Train Prediction Availability

- train_predictions_available: True

## Train Annotation Candidates

- sampled_train_clean_high_controls: 50
- sampled_train_high_false_positive_candidates: 80
- sampled_train_high_risk_low_candidates: 1
- sampled_train_mid_borderline_candidates: 27
- train_clean_high_controls: 2083
- train_high_false_positive_candidates: 479
- train_high_risk_low_candidates: 1
- train_mid_borderline_candidates: 27

## Dominant Metrics / Languages

- metrics: Basic Factual Accuracy=64, Content Relevance & Scope Control=21, Clarity, Simplicity & Inspiration=20, Domain Knowledge Accuracy=17, Error Identification & Correction Precision=14, Higher-Order Thinking & Skill Development=8, Motivation, Guidance & Positive Feedback=6, Instruction Following & Task Completion=5
- languages: en=87, zh=71

## Decision

- dev_audit_ready: True
- train_annotation_ready: True
- need_train_prediction_generation: False
- recommended_manual_annotation_count: 158
- next_step: `manual_annotation`

## Guardrails

- Test split is not read.
- No model training is performed.
- Dev cases are diagnostic only and must not be used as training labels.
- Full train prediction prompts/configs are generated under ignored output folders.
