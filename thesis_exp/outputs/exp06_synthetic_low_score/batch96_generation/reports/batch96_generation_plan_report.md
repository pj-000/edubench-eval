# Exp6-5 Batch96 Generation Plan Report

## Scope

This dry-run prepares the next 96-row question-disjoint low-score generation batch after manual
spot-check of the 17 filtered mini-batch samples. It does not train models, modify Exp0-Exp5
results, write checkpoints, call an API, or enter full 384-row generation.

## Curated Mini-batch

- Total filtered mini-batch rows: **17**
- Curated usable count: **16**
- Revised count: **3**
- Rejected count: **1**
- `mb008` enters low-score curated pool: **NO**

Curated files:

- `thesis_exp/outputs/exp06_synthetic_low_score/batch96_generation/curated/curated_mini_batch_synthetic_candidates.jsonl`
- `thesis_exp/outputs/exp06_synthetic_low_score/batch96_generation/curated/curated_mini_batch_revision_log.csv`
- `thesis_exp/outputs/exp06_synthetic_low_score/batch96_generation/curated/curated_mini_batch_rejected_or_relabel_upward.csv`

## Prompt and Filter Hardening

- Prompt hardening status: **PASS**
- Filter hardening status: **PASS**
- Hardened filter fields: `label_plausibility_status, error_type_alignment_status, rubric_failure_visibility, too_good_for_target_label, artifact_phrase_status, manual_review_required`

Prompt templates:

- `thesis_exp/outputs/exp06_synthetic_low_score/batch96_generation/prompt_templates/generate_low_score_answer_hardened.md`
- `thesis_exp/outputs/exp06_synthetic_low_score/batch96_generation/prompt_templates/generate_low_score_answer_hardened_en.md`
- `thesis_exp/outputs/exp06_synthetic_low_score/batch96_generation/prompt_templates/generate_low_score_answer_hardened_zh.md`

## Batch96 Plan

- Planned rows: **96**
- Prompt rows: **96**
- Target label distribution: `{'1': 40, '2': 40, '3': 16}`
- Language distribution: `{'en': 48, 'zh': 48}`
- Metric coverage: **12**
- Error type coverage: **7**
- Source split: **question_seed42/train only**
- Dev/test source overlap rows: **0**
- API called: **NO**
- Synthetic generated: **NO**

| synthetic_plan_id | target_label_5 | language | metric_canonical | error_type |
| --- | --- | --- | --- | --- |
| b96_001 | 1 | en | Reasoning Process Rigor | reasoning_gap |
| b96_002 | 1 | zh | Reasoning Process Rigor | overconfident_wrong |
| b96_003 | 1 | en | Reasoning Process Rigor | reasoning_gap |
| b96_004 | 1 | zh | Reasoning Process Rigor | overconfident_wrong |
| b96_005 | 2 | en | Reasoning Process Rigor | reasoning_gap |
| b96_006 | 2 | zh | Reasoning Process Rigor | overconfident_wrong |
| b96_007 | 2 | en | Reasoning Process Rigor | reasoning_gap |
| b96_008 | 3 | zh | Reasoning Process Rigor | overconfident_wrong |
| b96_009 | 1 | en | Scenario Element Integration | scenario_mismatch |
| b96_010 | 1 | zh | Scenario Element Integration | instruction_violation |
| b96_011 | 1 | en | Scenario Element Integration | scenario_mismatch |
| b96_012 | 1 | zh | Scenario Element Integration | instruction_violation |
| b96_013 | 2 | en | Scenario Element Integration | scenario_mismatch |
| b96_014 | 2 | zh | Scenario Element Integration | instruction_violation |
| b96_015 | 2 | en | Scenario Element Integration | scenario_mismatch |
| b96_016 | 3 | zh | Scenario Element Integration | instruction_violation |
| b96_017 | 1 | en | Personalization, Adaptation & Learning Support | scenario_mismatch |
| b96_018 | 1 | zh | Personalization, Adaptation & Learning Support | rubric_violation |
| b96_019 | 1 | en | Personalization, Adaptation & Learning Support | scenario_mismatch |
| b96_020 | 1 | zh | Personalization, Adaptation & Learning Support | rubric_violation |
| b96_021 | 2 | en | Personalization, Adaptation & Learning Support | scenario_mismatch |
| b96_022 | 2 | zh | Personalization, Adaptation & Learning Support | rubric_violation |
| b96_023 | 2 | en | Personalization, Adaptation & Learning Support | scenario_mismatch |
| b96_024 | 3 | zh | Personalization, Adaptation & Learning Support | rubric_violation |

_Showing 24 of 96 rows._

## Gates

- Can batch96 generation start? **YES**
- Can full 384 generation start? **NO**
- Can Exp6 training start? **NO**
