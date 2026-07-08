# Exp27I Target-Aware Teacher-Audit Collection

This collection summarizes parsed teacher outputs only. Raw API outputs remain ignored and must not
be committed.

## API Completion

- qwen/blind: rows=361, schema_ok=361, failed=0
- qwen/audit: rows=361, schema_ok=361, failed=0
- deepseek/blind: rows=361, schema_ok=361, failed=0
- deepseek/audit: rows=361, schema_ok=361, failed=0

## Target Scope

- qwen/blind: expected_target=361, unexpected_target=0, possible=15, high=
- qwen/audit: expected_target=361, unexpected_target=0, possible=, high=
- deepseek/blind: expected_target=361, unexpected_target=0, possible=3, high=
- deepseek/audit: expected_target=361, unexpected_target=0, possible=, high=

## Provider vs Original Human Label

- qwen: MAE=0.7867036011080333, bias=0.0332409972299169, low-human teacher-high=21, high-human teacher-low=18
- deepseek: MAE=0.853185595567867, bias=0.01662049861495845, low-human teacher-high=16, high-human teacher-low=22

## Conflict Queue

- conflict rows: 209
- top80_for_codex_direct_review: 80

Next step: Codex directly reviews the top conflicts by reading the actual context, evaluator output,
Qwen output, DeepSeek output, and original human label.
