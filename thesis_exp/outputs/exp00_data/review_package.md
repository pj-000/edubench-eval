# Exp 0 Review Package

## Key Output Files

- `thesis_exp/data/processed/edubench_scoring_all.jsonl`
- `thesis_exp/data/splits/paper_like_triple_seed42/`
- `thesis_exp/data/splits/question_seed42/`
- `thesis_exp/outputs/exp00_data/data_card.md`
- `thesis_exp/outputs/exp00_data/leakage_report.md`
- `thesis_exp/outputs/exp00_data/report.md`
- `thesis_exp/outputs/exp00_data/figures/`
- `thesis_exp/outputs/exp00_data/tables/`

## Key Counts

| item | value |
| --- | --- |
| main_dataset_rows | 5536 |
| paper_like_triple_seed42 | {"dev": 664, "test": 2218, "train": 2654} |
| question_seed42 | {"dev": 1107, "test": 1103, "train": 3326} |
| main_split_name | paper_like_triple_seed42 |
| leakage_status | WARNING |
| unmapped_metrics | 11 |
| unmapped_scenarios | 0 |
| invalid_or_ambiguous_scores | 0 |

## Five Standardized Sample Records

| record_id | metric | scenario | label_5 | question_preview |
| --- | --- | --- | --- | --- |
| 22fc3538594bd7f6426d96df9de31645e099e5df | Instruction Following & Task Completion | Personalized Content Creation | 4 | {'Name': 'Alex Johnson', 'Age': 20, 'Current Skill Level': 'Intermediate', 'Learning Goals': ['Understand cellular bi... |
| 00f03652813873e800443b4e84091ed05063eb49 | Scenario Element Integration | Personalized Content Creation | 4 | {'Name': 'Alex Johnson', 'Age': 20, 'Current Skill Level': 'Intermediate', 'Learning Goals': ['Understand cellular bi... |
| 51e5f21cbcf3748b0c03279f49b625af32cd9b0b | Personalization, Adaptation & Learning Support | Personalized Content Creation | 4 | {'Name': 'Alex Johnson', 'Age': 20, 'Current Skill Level': 'Intermediate', 'Learning Goals': ['Understand cellular bi... |
| a946b20864b31d0ab6a9e953e7ef387da3543466 | Instruction Following & Task Completion | Personalized Content Creation | 4 | {'Name': 'Alex Johnson', 'Age': 19, 'Current Education Level': 'Undergraduate, Year 1', 'Current Skill Level': 'Intro... |
| 605a39edf2ab4a11d8237e2d71382f0ab7c40100 | Scenario Element Integration | Personalized Content Creation | 4 | {'Name': 'Alex Johnson', 'Age': 19, 'Current Education Level': 'Undergraduate, Year 1', 'Current Skill Level': 'Intro... |

## Questions Requiring Human Confirmation

- Confirm that `results_merge.jsonl` is the intended final merged human-scored source for thesis experiments.
- `EduBench.zip` in this repository appears to contain paper/template assets rather than raw benchmark data; official scenario/metric alignment therefore uses the project definition and local `metrics_map.json`.
- Subject and education-level fields are inferred when explicit structured fields are absent, so these canonical values should be treated as audit metadata rather than gold labels.
