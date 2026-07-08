# Exp27C Teacher Audit V3 API Re-Pilot Report

This report summarizes the 60-row dual-teacher v3 re-pilot. It reads parsed outputs only.

## Decision

- recommendation: `revise_schema_or_prompt_before_361`
- proceed_to_361: False

## Core Metrics

- parse_success_rate: 1.0000
- schema_validation_success_rate: 1.0000
- semantic_validation_error_count: 14
- hard_validation_error_count: 6
- hard_validation_error_rate: 0.0250
- soft_validation_error_count: 8
- warning_count: 0
- usable_annotation_rate: 0.9750
- schema_repair_attempt_rate: 0.0042
- repair_success_rate: 1.0000
- repair_changed_teacher_score_count: 0
- repair_changed_major_failures_count: 0
- teacher_score_exact_agreement: 0.6333
- teacher_score_adjacent_agreement: 0.9167
- failure_visibility_exact_agreement: 0.6833
- overestimation_risk_exact_agreement: 0.6000
- evidence_type_exact_agreement: 0.7333
- major_failures_jaccard_mean: 0.7472
- evidence_span_valid_or_null_with_reason_rate: 0.9750
- label_quality_exact_agreement: 0.6333
- recommended_training_use_compatible_agreement: 0.9667
- needs_human_review_agreement: 0.7833
- high_control_suspected_conflict_rate: 0.1579

## Validation Error Breakdown

- deepseek/audit / hard / semantic_hard_rule: 1
- deepseek/audit / soft / reason_restates_score: 3
- deepseek/blind / hard / semantic_hard_rule: 1
- deepseek/blind / soft / reason_restates_score: 3
- qwen/audit / hard / evidence_span_invalid: 2
- qwen/audit / soft / reason_restates_score: 1
- qwen/blind / hard / evidence_span_invalid: 2
- qwen/blind / soft / reason_restates_score: 1

## Guardrails

- no training
- no GPU
- no test labels
- raw API outputs remain under ignored `annotations/raw_api/`
- full parsed teacher text remains under ignored `annotations/parsed/`
