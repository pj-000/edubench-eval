# Exp6-2 Train-only Synthetic Low-score Generation Plan

## Scope

This scaffold prepares train-only low-score synthetic generation with planned model
`deepseek-v4-pro`. It does **not** call APIs, generate synthetic data, train models, or
modify Exp0-Exp5 results.

## Why New Generation Is Needed

Exp6-0 found existing synthetic/sampled data mostly blocked by dev/test leakage, judge-output risk,
or unclear provenance. Exp6-1 found processed Excel candidates have model/pseudo labels and only 8
low-score train-only rows. Therefore a new train-only generation plan is needed before any Exp6
augmentation training.

## Source Sampling

Source rows come only from `paper_like_triple_seed42/train.jsonl`. Dev/test questions are not used.

- Sampling plan rows: **2654**
- Selected train source anchors: **71**
- Selected anchors cover metrics/languages from train only.

| source_record_id | metric_canonical | language | current_label_5 | planned_error_types | planned_target_labels | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 016540423ea53b5e0973f79a57e2d4cb12777b0c | Basic Factual Accuracy | en | 5 | ["factual_error", "overconfident_wrong", "reasoning_gap"] | [1, 2, 3] | train-only source; metric/language train low-count=2; budget=3 |
| 02890e57b4a5f86c993ca6597fa655df630aa99b | Basic Factual Accuracy | en | 5 | ["factual_error", "overconfident_wrong", "reasoning_gap"] | [1, 2, 3] | train-only source; metric/language train low-count=2; budget=3 |
| 0608f6457af621886d83afc93ddeb63b942c8c54 | Basic Factual Accuracy | en | 5 | ["factual_error", "overconfident_wrong", "reasoning_gap"] | [1, 2, 3] | train-only source; metric/language train low-count=2; budget=3 |
| 03755d25914c85c6eaabd46795cf656ef79566af | Basic Factual Accuracy | zh | 5 | ["factual_error", "overconfident_wrong", "reasoning_gap"] | [1, 2, 3] | train-only source; metric/language train low-count=4; budget=2 |
| 08112d7136d398859c6e989df0ead53ddc3c0e90 | Basic Factual Accuracy | zh | 5 | ["factual_error", "overconfident_wrong", "reasoning_gap"] | [1, 2, 3] | train-only source; metric/language train low-count=4; budget=2 |
| 042b1de277caa909648c04be2131a99e62c6b1fb | Clarity, Simplicity & Inspiration | en | 5 | ["superficial_fluency", "rubric_violation", "scenario_mismatch"] | [1, 2, 3] | train-only source; metric/language train low-count=3; budget=3 |
| 103e013057186f2db4db80cc42f2870441fa9328 | Clarity, Simplicity & Inspiration | en | 5 | ["superficial_fluency", "rubric_violation", "scenario_mismatch"] | [1, 2, 3] | train-only source; metric/language train low-count=3; budget=3 |
| 12e7fc799400eba02f528f565af32a0a0297d624 | Clarity, Simplicity & Inspiration | en | 5 | ["superficial_fluency", "rubric_violation", "scenario_mismatch"] | [1, 2, 3] | train-only source; metric/language train low-count=3; budget=3 |
| 02072d13e99d6ee933d8a792fccced69a4c70245 | Clarity, Simplicity & Inspiration | zh | 5 | ["superficial_fluency", "rubric_violation", "scenario_mismatch"] | [1, 2, 3] | train-only source; metric/language train low-count=0; budget=4 |
| 0bb74dfb863e3e3d4689c23feff099a6b7fd0ec2 | Clarity, Simplicity & Inspiration | zh | 5 | ["superficial_fluency", "rubric_violation", "scenario_mismatch"] | [1, 2, 3] | train-only source; metric/language train low-count=0; budget=4 |
| 101c5bb5890cb28a47d4f44d565ef0a7bc567673 | Clarity, Simplicity & Inspiration | zh | 5 | ["superficial_fluency", "rubric_violation", "scenario_mismatch"] | [1, 2, 3] | train-only source; metric/language train low-count=0; budget=4 |
| 13e30f6f382a157140477a7aa3272b65bd2f4039 | Clarity, Simplicity & Inspiration | zh | 5 | ["superficial_fluency", "rubric_violation", "scenario_mismatch"] | [1, 2, 3] | train-only source; metric/language train low-count=0; budget=4 |
| 0323ddc9b4a1094a44e0b027e36cd8e15d76f93c | Content Relevance & Scope Control | en | 5 | ["scenario_mismatch", "rubric_violation", "superficial_fluency"] | [1, 2, 3] | train-only source; metric/language train low-count=2; budget=3 |
| 03a0da37856d7f5174ddbd5db28290e222c32335 | Content Relevance & Scope Control | en | 5 | ["scenario_mismatch", "rubric_violation", "superficial_fluency"] | [1, 2, 3] | train-only source; metric/language train low-count=2; budget=3 |
| 03f5d9758c7a62f42ea24d2e10cba7cafbd492f4 | Content Relevance & Scope Control | en | 5 | ["scenario_mismatch", "rubric_violation", "superficial_fluency"] | [1, 2, 3] | train-only source; metric/language train low-count=2; budget=3 |
| 0029a0a8d467908b7c7bfc7302845f54df20a94e | Content Relevance & Scope Control | zh | 5 | ["scenario_mismatch", "rubric_violation", "superficial_fluency"] | [1, 2, 3] | train-only source; metric/language train low-count=2; budget=3 |
| 05719e6a594b02067f38995842cea54ddd2e2bcc | Content Relevance & Scope Control | zh | 5 | ["scenario_mismatch", "rubric_violation", "superficial_fluency"] | [1, 2, 3] | train-only source; metric/language train low-count=2; budget=3 |
| 0795911801d8484e3269e88e83494fe544417969 | Content Relevance & Scope Control | zh | 5 | ["scenario_mismatch", "rubric_violation", "superficial_fluency"] | [1, 2, 3] | train-only source; metric/language train low-count=2; budget=3 |
| 0b0d3c82292e8d3242c48804bf755fd31505d240 | Domain Knowledge Accuracy | en | 5 | ["factual_error", "overconfident_wrong", "rubric_violation"] | [1, 2, 3] | train-only source; metric/language train low-count=0; budget=4 |
| 16d03816c40a0902e7a8a05c1cefbb887b75d35b | Domain Knowledge Accuracy | en | 5 | ["factual_error", "overconfident_wrong", "rubric_violation"] | [1, 2, 3] | train-only source; metric/language train low-count=0; budget=4 |
| 19eb8738e781f5ecc14242215823ee93e415820c | Domain Knowledge Accuracy | en | 5 | ["factual_error", "overconfident_wrong", "rubric_violation"] | [1, 2, 3] | train-only source; metric/language train low-count=0; budget=4 |
| 1ffc8d3c3c3ca3a05f9b1384b40907d83712febd | Domain Knowledge Accuracy | en | 5 | ["factual_error", "overconfident_wrong", "rubric_violation"] | [1, 2, 3] | train-only source; metric/language train low-count=0; budget=4 |
| 02789727b2e117e08a4e240edba2522b63e4488e | Domain Knowledge Accuracy | zh | 5 | ["factual_error", "overconfident_wrong", "rubric_violation"] | [1, 2, 3] | train-only source; metric/language train low-count=0; budget=4 |
| 0284805df66b2a4fdfe8be98139ba1385f7021bc | Domain Knowledge Accuracy | zh | 5 | ["factual_error", "overconfident_wrong", "rubric_violation"] | [1, 2, 3] | train-only source; metric/language train low-count=0; budget=4 |

_Showing 24 of 71 rows._

## Error Types

| error_type | target_label_range | applicable_metrics | expected_risk |
| --- | --- | --- | --- |
| factual_error | 1-2 | ["Basic Factual Accuracy", "Domain Knowledge Accuracy", "Error Identi... | May accidentally produce a correct answer; needs optional judge/human... |
| reasoning_gap | 1-3 | ["Basic Factual Accuracy", "Error Identification & Correction Precisi... | May be too subtle for label 1; label 2/3 safer. |
| instruction_violation | 1-2 | ["Instruction Following & Task Completion", "Personalization, Adaptat... | Can create obvious artifacts; filter for naturalness. |
| scenario_mismatch | 1-3 | ["Clarity, Simplicity & Inspiration", "Content Relevance & Scope Cont... | Might affect multiple metrics; record target metric explicitly. |
| rubric_violation | 1-2 | ["Clarity, Simplicity & Inspiration", "Content Relevance & Scope Cont... | Requires rubric quality; skip if rubric is missing. |
| superficial_fluency | 2-3 | ["Clarity, Simplicity & Inspiration", "Content Relevance & Scope Cont... | Often label 3 rather than 1/2; use for boundary negatives. |
| overconfident_wrong | 1-2 | ["Basic Factual Accuracy", "Domain Knowledge Accuracy", "Error Identi... | Can be too adversarial; keep pedagogically plausible. |

## Generation Target Matrix

First low-score augmentation target: **384** rows.

Target label counts: `{'1': 168, '2': 168, '3': 48}`

The first matrix emphasizes labels 1/2 with a small label-3 boundary set. For D1/D4 synthetic-only
diagnostics, `synthetic_only_diagnostic_target_matrix.csv` provides an optional full-score matrix
with labels 1-5, but it is not part of the first low-score augmentation generation.

## Prompt Templates

- `thesis_exp/outputs/exp06_synthetic_low_score/generation_plan/prompt_templates/generate_low_score_answer.md`
- `thesis_exp/outputs/exp06_synthetic_low_score/generation_plan/prompt_templates/generate_low_score_answer_en.md`
- `thesis_exp/outputs/exp06_synthetic_low_score/generation_plan/prompt_templates/generate_low_score_answer_zh.md`
- `thesis_exp/outputs/exp06_synthetic_low_score/generation_plan/samples/dry_run_prompts.jsonl`

Prompts instruct the model to generate natural flawed answers without mentioning synthetic or
intentional wrongness. They request JSON output but do not call any API.

## Schema, Filtering, Leakage

- `thesis_exp/outputs/exp06_synthetic_low_score/generation_plan/synthetic_schema.md`
- `thesis_exp/outputs/exp06_synthetic_low_score/generation_plan/tables/filtering_rules.csv`
- `thesis_exp/outputs/exp06_synthetic_low_score/generation_plan/leakage_check_plan.md`

Required checks include valid JSON, non-empty answer, length bounds, language match, no explicit
intentional-wrong phrasing, no copied original answer, valid label/metric/rubric, source split
train,
deduplication, and no dev/test leakage.

## Recommendation

Generation can start only after human review approves the prompt templates, target matrix, API
budget, and leakage/filtering workflow. Training still cannot start until generated rows pass
schema,
filtering, dedup, and dev/test leakage checks.
