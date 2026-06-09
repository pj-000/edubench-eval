# Exp6-6 Batch96 Leakage Report

Status: **PASS**

- Filtered rows checked: **91**
- Blocked rows: **0**
- Leakage summary: `thesis_exp/outputs/exp06_synthetic_low_score/batch96_generation/leakage/leakage_summary.csv`
- Leakage details: `thesis_exp/outputs/exp06_synthetic_low_score/batch96_generation/leakage/leakage_details.csv`

Checks:

- source_question_key not in dev/test
- source_triple_key not in dev/test
- synthetic question not in dev/test
- synthetic question+answer not in dev/test
- answer_synthetic not duplicate with human test answer
- duplicate synthetic answer
