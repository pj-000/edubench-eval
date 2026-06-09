# Exp6-11 Topup-2 Leakage Report

Status: **PASS**

- Filtered rows checked: **157**
- Blocked rows: **0**
- Leakage summary: `thesis_exp/outputs/exp06_synthetic_low_score/topup_generation/topup2/leakage/leakage_summary.csv`
- Leakage details: `thesis_exp/outputs/exp06_synthetic_low_score/topup_generation/topup2/leakage/leakage_details.csv`

Checks:

- source_question_key not in dev/test
- source_triple_key not in dev/test
- synthetic question not in dev/test
- synthetic question+answer not in dev/test
- answer_synthetic not duplicate with human test answer
- duplicate synthetic answer
