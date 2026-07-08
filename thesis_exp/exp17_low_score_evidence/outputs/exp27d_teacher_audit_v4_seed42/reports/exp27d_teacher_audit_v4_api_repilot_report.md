# Exp27D Teacher Audit V4 API Re-Pilot Report

This report summarizes parsed teacher outputs only. Raw API and full parsed text stay ignored.

## Decision

- recommendation: `revise_schema_or_prompt_before_361`
- proceed_to_361: False

## Core Metrics

- parse_success_rate: 1.0000
- schema_validation_success_rate: 1.0000
- hard_validation_error_count: 8
- hard_validation_error_rate: 0.0250
- usable_annotation_rate: 0.9781
- repair_changed_judgement_count: 0
- teacher_score_exact_agreement: 0.6625
- teacher_score_adjacent_agreement: 0.8375
- failure_bucket_agreement: 0.7000
- derived_overestimation_risk_agreement: 0.7750
- evidence_span_valid_or_null_with_reason_rate: 0.9688
- recommended_training_use_compatible_agreement: 0.9375
- high_control_hard_conflict_rate: 0.1852

## Derived Risk Rule

- version: `exp27d_v4_rule_20260708`
- answer_key_uncertainty in {possible_answer_key_issue, insufficient_reference, rubric_ambiguous} -> unclear
- score_cap <= 2 -> high
- teacher_score <= 2 and failure_bucket=hidden_or_missing_failure and surface_plausibility in {high, medium} -> high
- teacher_score <= 2 and failure_bucket=hidden_or_missing_failure and surface_plausibility=low -> medium
- teacher_score <= 2 and failure_bucket=visible_failure -> medium
- teacher_score == 3 and failure_bucket=hidden_or_missing_failure and surface_plausibility=high -> medium
- teacher_score >= 4 and failure_bucket=no_failure -> low
- failure_bucket=unclear or confidence=low -> unclear
- otherwise -> low

## Thresholds

- parse >= 0.99
- schema >= 0.99
- usable >= 0.95
- hard error rate <= 0.03
- adjacent score agreement >= 0.95
- exact score agreement >= 0.60
- failure bucket agreement >= 0.80
- derived risk agreement >= 0.75
- evidence valid >= 0.95
- training-use compatible >= 0.95
- high-control hard conflict <= 0.10
- repair_changed_judgement_count == 0

## Guardrails

- no training
- no GPU
- no test labels
- D1/dev labels are not used
- raw API outputs remain under ignored `annotations/raw_api/`
- full parsed teacher text remains under ignored `annotations/parsed/`
