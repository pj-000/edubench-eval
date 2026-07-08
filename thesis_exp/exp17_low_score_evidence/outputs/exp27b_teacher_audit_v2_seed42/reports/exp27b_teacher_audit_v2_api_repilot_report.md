# Exp27B Teacher Audit V2 API Re-Pilot Report

This report summarizes the 60-row dual-teacher v2 re-pilot. It reads parsed outputs only.

## Decision

- recommendation: `revise_schema_or_prompt_before_361`
- proceed_to_361: False

## Core Metrics

- parse_success_rate: 1.0000
- schema_validation_success_rate: 0.9542
- semantic_validation_error_count: 87
- teacher_score_exact_agreement: 0.7333
- teacher_score_adjacent_agreement: 0.9167
- failure_visibility_exact_agreement: 0.7667
- overestimation_risk_exact_agreement: 0.6333
- evidence_type_exact_agreement: 0.7500
- major_failures_jaccard_mean: 0.6972
- evidence_span_valid_or_null_with_reason_rate: 0.9833
- label_quality_exact_agreement: 0.7333
- recommended_training_use_compatible_agreement: 0.9500
- needs_human_review_agreement: 0.8667
- high_control_suspected_conflict_rate: 0.1579

## Validation Error Breakdown

- deepseek/audit / audit_gap_review_flag: 1
- deepseek/audit / missing_score_cap_for_low_failure: 10
- deepseek/audit / no_major_failure_risk_not_low: 7
- deepseek/audit / reason_restates_score: 1
- deepseek/blind / missing_score_cap_for_low_failure: 10
- deepseek/blind / no_major_failure_risk_not_low: 7
- deepseek/blind / reason_restates_score: 1
- qwen/audit / evidence_span_not_substring: 1
- qwen/audit / no_major_failure_risk_not_low: 1
- qwen/audit / other: 3
- qwen/audit / reason_restates_score: 2
- qwen/audit / schema_error: 20
- qwen/blind / evidence_span_not_substring: 1
- qwen/blind / no_major_failure_risk_not_low: 1
- qwen/blind / other: 1
- qwen/blind / reason_restates_score: 2
- qwen/blind / schema_error: 18

## Guardrails

- no training
- no GPU
- no test labels
- raw API outputs remain under ignored `annotations/raw_api/`
- full parsed teacher text remains under ignored `annotations/parsed/`
