# Exp27C Teacher Audit V3 Validation

- packets: 60
- annotation_rows: 240
- errors: 14
- hard_errors: 6
- soft_errors: 8
- warnings: 0
- exp27a_overlap_present: 20/20
- schema_validation_passed: False
- ready_for_v3_api_repilot: False
- test_label_read: False

## First Errors

- deepseek/blind[12]: missing_required_content should use evidence_span=null
- deepseek/blind[38]: teacher_reason restates score field
- deepseek/blind[39]: teacher_reason restates score field
- deepseek/blind[54]: teacher_reason restates score field
- deepseek/audit[12]: missing_required_content should use evidence_span=null
- deepseek/audit[38]: teacher_reason restates score field
- deepseek/audit[39]: teacher_reason restates score field
- deepseek/audit[54]: teacher_reason restates score field
- qwen/blind[14]: evidence_span is not exact or normalized substring of answer
- qwen/blind[21]: evidence_span is not exact or normalized substring of answer
- qwen/blind[30]: teacher_reason restates score field
- qwen/audit[14]: evidence_span is not exact or normalized substring of answer
- qwen/audit[21]: evidence_span is not exact or normalized substring of answer
- qwen/audit[29]: teacher_reason restates score field
