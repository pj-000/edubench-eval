# Exp27G Teacher-Audited 361 Preparation

This step prepares a train-only 361-case teacher-audit expansion. It does not call APIs or train.

## Counts

- packets: 361
- batch_count: 19
- label_counts: `{1: 58, 2: 53, 3: 85, 4: 71, 5: 94}`
- group_counts: `{'exp27f_top40_conflict_reaudit': 40, 'train_all_low_label': 95, 'train_clean_high_controls': 36, 'train_education_dimension_stress': 40, 'train_high_disagreement_protection': 80, 'train_mid_borderline': 70}`

## Sampling Strategy

- Start with Exp27F top40 conflict cases for re-audit.
- Include all available train low-label cases after de-duplication.
- Add high-label disagreement cases for high-score protection.
- Add label-3 borderline cases.
- Add education/rubric-dimension stress cases.
- Fill remaining rows with clean high controls.

## Guardrails

- Blind packets contain no original score and no recovered human reason.
- Audit reference contains train-only original scores for audit stage.
- Dev/test are used only for sample_id/question_key leakage guards.
- Test labels are not read.
- No API call or model training is performed in preparation.
