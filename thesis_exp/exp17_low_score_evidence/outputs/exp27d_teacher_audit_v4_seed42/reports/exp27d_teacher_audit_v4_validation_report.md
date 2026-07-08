# Exp27D Teacher Audit V4 Validation

- packets: 80
- annotation_rows: 320
- errors: 14
- hard_errors: 8
- soft_errors: 6
- warnings: 0
- exp27c_overlap_present: 60/60
- schema_validation_passed: False
- ready_for_v4_api_repilot: False
- test_label_read: False

## First Errors

- deepseek/blind[6]: high no_major_failure must use overestimation_risk=low
- deepseek/blind[14]: evidence_span is not exact or normalized substring of answer
- deepseek/blind[21]: teacher_reason restates score field
- deepseek/blind[34]: teacher_reason restates score field
- deepseek/blind[45]: high no_major_failure must use overestimation_risk=low
- deepseek/blind[52]: teacher_reason restates score field
- deepseek/blind[54]: teacher_reason restates score field
- deepseek/blind[59]: teacher_reason restates score field
- deepseek/blind[61]: teacher_reason restates score field
- deepseek/audit[70]: gap>=2 should use needs_human_review=true
- deepseek/audit[70]: gap>=2 should use hard_conflict=true
- qwen/blind[14]: evidence_span is not exact or normalized substring of answer
- qwen/blind[21]: evidence_span is not exact or normalized substring of answer
- qwen/blind[26]: evidence_span is not exact or normalized substring of answer
