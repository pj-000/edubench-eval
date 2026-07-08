# Exp27G 361 Teacher-Audit Collection

This collection summarizes parsed teacher outputs only. Raw API outputs remain ignored and are not
intended for commit.

## API Completion

- qwen/blind: rows=361, schema_ok=361, failed=0
- qwen/audit: rows=361, schema_ok=361, failed=0
- deepseek/blind: rows=361, schema_ok=361, failed=0
- deepseek/audit: rows=361, schema_ok=361, failed=0

## Provider vs Original Human Label

- qwen: MAE=0.7811634349030471, bias=-0.08310249307479224, low-human teacher-high=17, high-human teacher-low=19
- deepseek: MAE=0.8781163434903048, bias=-0.1634349030470914, low-human teacher-high=12, high-human teacher-low=31

## Conflict Queue

- conflict rows: 206
- top80_for_adjudication: 80

Next step: adjudicate the top conflict queue by comparing original human label, Qwen label, and
DeepSeek label.
