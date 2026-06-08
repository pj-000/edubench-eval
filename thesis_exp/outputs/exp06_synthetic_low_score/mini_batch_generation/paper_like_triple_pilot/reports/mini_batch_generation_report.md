# Exp6-3b Mini-batch Plan: paper_like_triple_pilot

Mode purpose: high-risk prompt/debug pilot only

- Split dir: `thesis_exp/data/splits/paper_like_triple_seed42`
- Allowed for training after generation audit: **False**
- Risk level: **HIGH**
- API called: **NO**
- Synthetic answers generated: **NO**
- Planned rows: **24**
- Selected source rows: **24**
- Dry-run prompt rows: **24**
- Target label distribution: `{1: 10, 2: 10, 3: 4}`
- Language distribution: `{'en': 12, 'zh': 12}`
- Metric coverage: **12**
- Error type coverage: **7**
- Source question overlap rows selected: **24**
- Can mini-batch generation start after human review: **NO**
- Can full 384 generation start: **NO**
- Can Exp6 training start: **NO**

| synthetic_plan_id | source_record_id | language | metric_canonical | target_label_5 | error_type | source_risk_level | risk_notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mb001 | 0786532c4978fd30afcfa51102d5f96fd2d16425 | en | Reasoning Process Rigor | 1 | reasoning_gap | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb002 | 0ee29e83c6f2ac7784c585acebb931f7dc1b7dba | zh | Reasoning Process Rigor | 2 | overconfident_wrong | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb003 | 0204872bcf4cd4b4d4937c12c2cd77f0171376ab | en | Scenario Element Integration | 1 | scenario_mismatch | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb004 | 048058816bba705040007e7b0c822282f1c5defa | zh | Scenario Element Integration | 2 | instruction_violation | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb005 | 01ffcbf103c1dce7417ed4518a50fc19a77d1ee9 | en | Personalization, Adaptation & Learning Support | 1 | scenario_mismatch | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb006 | 1d591637c0cb75eefc0b66b1df765efdfbcf8bd4 | zh | Personalization, Adaptation & Learning Support | 2 | rubric_violation | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb007 | 0437d0cfa0dce66ccca098964dc8a1bba6a151b5 | en | Higher-Order Thinking & Skill Development | 1 | reasoning_gap | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb008 | 092e45579d1eb967e6b69d085b1d122398a4e99d | zh | Higher-Order Thinking & Skill Development | 2 | superficial_fluency | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb009 | 03c02b32f53105706e139a71aaa785183488cd6e | en | Error Identification & Correction Precision | 1 | factual_error | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb010 | 0b0b1d9ef4015668d3e17fe40a86c301dd6b40f3 | zh | Error Identification & Correction Precision | 2 | reasoning_gap | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb011 | 01294d6a09e14ab33102375daee613d5e0226caa | en | Instruction Following & Task Completion | 1 | instruction_violation | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb012 | 0079b701dcca5c05742158464604e17b2351f5a0 | zh | Instruction Following & Task Completion | 2 | rubric_violation | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb013 | 016540423ea53b5e0973f79a57e2d4cb12777b0c | en | Basic Factual Accuracy | 1 | factual_error | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb014 | 03755d25914c85c6eaabd46795cf656ef79566af | zh | Basic Factual Accuracy | 2 | overconfident_wrong | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb015 | 0b0d3c82292e8d3242c48804bf755fd31505d240 | en | Domain Knowledge Accuracy | 1 | factual_error | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb016 | 02789727b2e117e08a4e240edba2522b63e4488e | zh | Domain Knowledge Accuracy | 2 | rubric_violation | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb017 | 12326b370deba9d6576e667938f836f2c5fdbe73 | en | Role & Tone Consistency | 3 | scenario_mismatch | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb018 | 0112721b014104978a66accce072f2b6212d64c8 | zh | Role & Tone Consistency | 1 | superficial_fluency | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb019 | 0323ddc9b4a1094a44e0b027e36cd8e15d76f93c | en | Content Relevance & Scope Control | 3 | scenario_mismatch | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb020 | 0029a0a8d467908b7c7bfc7302845f54df20a94e | zh | Content Relevance & Scope Control | 1 | rubric_violation | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb021 | 042b1de277caa909648c04be2131a99e62c6b1fb | en | Clarity, Simplicity & Inspiration | 2 | superficial_fluency | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb022 | 02072d13e99d6ee933d8a792fccced69a4c70245 | zh | Clarity, Simplicity & Inspiration | 3 | rubric_violation | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb023 | 0747575357db0dee2bc1b170eca12dbb572f2be3 | en | Motivation, Guidance & Positive Feedback | 2 | scenario_mismatch | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
| mb024 | 0347526cae737a75caf2cfaa5723714c406643db | zh | Motivation, Guidance & Positive Feedback | 3 | superficial_fluency | HIGH | HIGH_RISK_QUESTION_OVERLAP; prompt/debug only; not allowed for training. |
