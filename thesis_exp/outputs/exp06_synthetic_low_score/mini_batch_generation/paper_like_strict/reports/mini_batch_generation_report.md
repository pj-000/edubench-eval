# Exp6-3b Mini-batch Plan: paper_like_strict

Mode purpose: demonstrate strict source unavailability under paper-like split

- Split dir: `thesis_exp/data/splits/paper_like_triple_seed42`
- Allowed for training after generation audit: **False**
- Risk level: **BLOCKED**
- API called: **NO**
- Synthetic answers generated: **NO**
- Planned rows: **24**
- Selected source rows: **0**
- Dry-run prompt rows: **0**
- Target label distribution: `{1: 10, 2: 10, 3: 4}`
- Language distribution: `{'en': 12, 'zh': 12}`
- Metric coverage: **12**
- Error type coverage: **7**
- Source question overlap rows selected: **0**
- Can mini-batch generation start after human review: **NO**
- Can full 384 generation start: **NO**
- Can Exp6 training start: **NO**

| synthetic_plan_id | source_record_id | language | metric_canonical | target_label_5 | error_type | source_risk_level | risk_notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mb001 |  | en | Reasoning Process Rigor | 1 | reasoning_gap | BLOCKED | No eligible source under this split mode. |
| mb002 |  | zh | Reasoning Process Rigor | 2 | overconfident_wrong | BLOCKED | No eligible source under this split mode. |
| mb003 |  | en | Scenario Element Integration | 1 | scenario_mismatch | BLOCKED | No eligible source under this split mode. |
| mb004 |  | zh | Scenario Element Integration | 2 | instruction_violation | BLOCKED | No eligible source under this split mode. |
| mb005 |  | en | Personalization, Adaptation & Learning Support | 1 | scenario_mismatch | BLOCKED | No eligible source under this split mode. |
| mb006 |  | zh | Personalization, Adaptation & Learning Support | 2 | rubric_violation | BLOCKED | No eligible source under this split mode. |
| mb007 |  | en | Higher-Order Thinking & Skill Development | 1 | reasoning_gap | BLOCKED | No eligible source under this split mode. |
| mb008 |  | zh | Higher-Order Thinking & Skill Development | 2 | superficial_fluency | BLOCKED | No eligible source under this split mode. |
| mb009 |  | en | Error Identification & Correction Precision | 1 | factual_error | BLOCKED | No eligible source under this split mode. |
| mb010 |  | zh | Error Identification & Correction Precision | 2 | reasoning_gap | BLOCKED | No eligible source under this split mode. |
| mb011 |  | en | Instruction Following & Task Completion | 1 | instruction_violation | BLOCKED | No eligible source under this split mode. |
| mb012 |  | zh | Instruction Following & Task Completion | 2 | rubric_violation | BLOCKED | No eligible source under this split mode. |
| mb013 |  | en | Basic Factual Accuracy | 1 | factual_error | BLOCKED | No eligible source under this split mode. |
| mb014 |  | zh | Basic Factual Accuracy | 2 | overconfident_wrong | BLOCKED | No eligible source under this split mode. |
| mb015 |  | en | Domain Knowledge Accuracy | 1 | factual_error | BLOCKED | No eligible source under this split mode. |
| mb016 |  | zh | Domain Knowledge Accuracy | 2 | rubric_violation | BLOCKED | No eligible source under this split mode. |
| mb017 |  | en | Role & Tone Consistency | 3 | scenario_mismatch | BLOCKED | No eligible source under this split mode. |
| mb018 |  | zh | Role & Tone Consistency | 1 | superficial_fluency | BLOCKED | No eligible source under this split mode. |
| mb019 |  | en | Content Relevance & Scope Control | 3 | scenario_mismatch | BLOCKED | No eligible source under this split mode. |
| mb020 |  | zh | Content Relevance & Scope Control | 1 | rubric_violation | BLOCKED | No eligible source under this split mode. |
| mb021 |  | en | Clarity, Simplicity & Inspiration | 2 | superficial_fluency | BLOCKED | No eligible source under this split mode. |
| mb022 |  | zh | Clarity, Simplicity & Inspiration | 3 | rubric_violation | BLOCKED | No eligible source under this split mode. |
| mb023 |  | en | Motivation, Guidance & Positive Feedback | 2 | scenario_mismatch | BLOCKED | No eligible source under this split mode. |
| mb024 |  | zh | Motivation, Guidance & Positive Feedback | 3 | superficial_fluency | BLOCKED | No eligible source under this split mode. |
