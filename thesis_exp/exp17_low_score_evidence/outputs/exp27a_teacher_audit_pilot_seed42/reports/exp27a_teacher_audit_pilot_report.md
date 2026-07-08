# Exp27A Teacher Audit Pilot Packets

This step prepares a two-stage teacher-audited annotation pilot. It does not call any model.

## Counts

- pilot rows: 361
- train low rows: 111
- train mid sampled rows: 100
- train high-control sampled rows: 150
- batches: 19

## Guardrails

- Blind packet prompts do not expose the original score to the teacher.
- Original scores for audit are stored only in `packets/exp27a_pilot_audit_reference_private.jsonl`.
- Dev/test are read only for sample_id/question_key leakage guards.
- Test labels are not read.
- Teacher output is an audit signal, not a replacement gold label.
