# Exp27I Target-Aware Teacher-Audit Preparation

Exp27I reuses the Exp27G train-only 361 sample set, but fixes the scoring target ambiguity.

## Target Rule

- The `question` field is wrapped as context only.
- The `answer` field is wrapped as the evaluator output to score.
- Blind and audit schemas require explicit target-scope fields.
- The API stage still does not see dev/test labels or recovered human reasons.

## Counts

- packets: 361
- label_counts: `{1: 58, 2: 53, 3: 85, 4: 71, 5: 94}`
- group_counts: `{'exp27f_top40_conflict_reaudit': 40, 'train_all_low_label': 95, 'train_clean_high_controls': 36, 'train_education_dimension_stress': 40, 'train_high_disagreement_protection': 80, 'train_mid_borderline': 70}`

## Next Step

Run Qwen and DeepSeek on these target-aware packets, then perform direct Codex semantic adjudication
on teacher/human conflicts.
