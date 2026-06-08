# Exp6-3b Mini-batch Plan: question_disjoint_formal

Mode purpose: formal leakage-safe synthetic generation and augmentation

- Split dir: `thesis_exp/data/splits/question_seed42`
- Allowed for training after generation audit: **True**
- Risk level: **LOW**
- API called: **NO**
- Synthetic answers generated: **NO**
- Planned rows: **24**
- Selected source rows: **24**
- Dry-run prompt rows: **24**
- Target label distribution: `{1: 10, 2: 10, 3: 4}`
- Language distribution: `{'en': 12, 'zh': 12}`
- Metric coverage: **12**
- Error type coverage: **7**
- Source question overlap rows selected: **0**
- Can mini-batch generation start after human review: **YES**
- Can full 384 generation start: **NO**
- Can Exp6 training start: **NO**

| synthetic_plan_id | source_record_id | language | metric_canonical | target_label_5 | error_type | source_risk_level | risk_notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mb001 | 0786532c4978fd30afcfa51102d5f96fd2d16425 | en | Reasoning Process Rigor | 1 | reasoning_gap | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb002 | 0ee29e83c6f2ac7784c585acebb931f7dc1b7dba | zh | Reasoning Process Rigor | 2 | overconfident_wrong | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb003 | 0204872bcf4cd4b4d4937c12c2cd77f0171376ab | en | Scenario Element Integration | 1 | scenario_mismatch | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb004 | 048058816bba705040007e7b0c822282f1c5defa | zh | Scenario Element Integration | 2 | instruction_violation | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb005 | 01ffcbf103c1dce7417ed4518a50fc19a77d1ee9 | en | Personalization, Adaptation & Learning Support | 1 | scenario_mismatch | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb006 | 04d76b45dda841d0666b8746dc95021a138ec9c4 | zh | Personalization, Adaptation & Learning Support | 2 | rubric_violation | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb007 | 0437d0cfa0dce66ccca098964dc8a1bba6a151b5 | en | Higher-Order Thinking & Skill Development | 1 | reasoning_gap | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb008 | 0b24303ee333541742acab0e6ac342abe677c221 | zh | Higher-Order Thinking & Skill Development | 2 | superficial_fluency | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb009 | 000b1f47c21ee93516a424f0cba80a7188cf3e6f | en | Error Identification & Correction Precision | 1 | factual_error | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb010 | 0b0b1d9ef4015668d3e17fe40a86c301dd6b40f3 | zh | Error Identification & Correction Precision | 2 | reasoning_gap | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb011 | 01294d6a09e14ab33102375daee613d5e0226caa | en | Instruction Following & Task Completion | 1 | instruction_violation | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb012 | 017b36d1dcb5ac4eb79d1b467f6bf1403d6b7fef | zh | Instruction Following & Task Completion | 2 | rubric_violation | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb013 | 014fc19dc4186ea41ccc098771dd4202c2d97031 | en | Basic Factual Accuracy | 1 | factual_error | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb014 | 01aa6c4db3c46a733791bd16f3386308ac5a4b7b | zh | Basic Factual Accuracy | 2 | overconfident_wrong | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb015 | 0b0d3c82292e8d3242c48804bf755fd31505d240 | en | Domain Knowledge Accuracy | 1 | factual_error | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb016 | 0284805df66b2a4fdfe8be98139ba1385f7021bc | zh | Domain Knowledge Accuracy | 2 | rubric_violation | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb017 | 0362a81a2b2675a908b5f959600c1d353d74b7ba | en | Role & Tone Consistency | 3 | scenario_mismatch | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb018 | 203522e6df4c5c5a7fc4d604858ea8b47f105be6 | zh | Role & Tone Consistency | 1 | superficial_fluency | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb019 | 0323ddc9b4a1094a44e0b027e36cd8e15d76f93c | en | Content Relevance & Scope Control | 3 | scenario_mismatch | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb020 | 0029a0a8d467908b7c7bfc7302845f54df20a94e | zh | Content Relevance & Scope Control | 1 | rubric_violation | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb021 | 042b1de277caa909648c04be2131a99e62c6b1fb | en | Clarity, Simplicity & Inspiration | 2 | superficial_fluency | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb022 | 033c60ce75db4441ba76414ef486a2b23cce4daf | zh | Clarity, Simplicity & Inspiration | 3 | rubric_violation | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb023 | 0747575357db0dee2bc1b170eca12dbb572f2be3 | en | Motivation, Guidance & Positive Feedback | 2 | scenario_mismatch | LOW | question-disjoint formal source; eligible for generation after prompt... |
| mb024 | 07a9883089c759edc2978f3ee1f4bf2c3cad79f3 | zh | Motivation, Guidance & Positive Feedback | 3 | superficial_fluency | LOW | question-disjoint formal source; eligible for generation after prompt... |
