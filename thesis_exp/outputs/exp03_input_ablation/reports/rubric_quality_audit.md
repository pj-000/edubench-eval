# Exp3 Rubric Quality Audit

Overall status: **ERROR**

- Input: `thesis_exp/outputs/exp03_input_ablation/tables/rubric_source_audit.csv`
- Pairwise comparisons: 132
- ERROR pairs: 1
- WARNING pairs: 0
- zh Scenario Element Integration vs Instruction Following & Task Completion: **ERROR**

Exact duplicates across different metrics are marked ERROR. Highly similar cross-metric
rubrics are marked WARNING at similarity >= 0.90.

CSV: `thesis_exp/outputs/exp03_input_ablation/tables/rubric_quality_audit.csv`

## ERROR Pairs

| language | metric_a | metric_b | similarity | notes |
| --- | --- | --- | ---: | --- |
| zh | Instruction Following & Task Completion | Scenario Element Integration | 1.000000 | SPECIAL_CHECK zh Scenario Element Integration vs Instruction Following & Task Completion. Different metrics have exactly identical rubric text. |
