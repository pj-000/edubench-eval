# Exp19-R6 Real DPO Dataset Report

Exp19-R6 builds DPO preference pairs from train split only. The rejected side is always a
real score emitted by an existing judge or by Qwen3-4B direct baseline; no template or hard
synthetic rejected response is used.

## Construction Rule

- input prompt: question + answer + metric + rubric + metadata.
- chosen response: human gold score, optionally with structured hidden-failure fields.
- rejected response: real wrong model score from train-side reusable judge outputs.
- strict main variants are score-only because reusable real rejected outputs are scalar scores.
- structured/reason chosen variants are auxiliary review artifacts; their rejected side remains score-only.

## Guardrails

- test split is not read.
- dev/D1 annotations are not used for training labels.
- human rationale is not used as prompt input.
- rejected responses are not handwritten templates.
- full R6 DPO JSON is written under `data/` and committed when explicitly requested for external review.

## Train / Pair Counts

- train samples: 3326
- unique real rejected pair pool: 1250
- high_to_low_real_model_error: 224
- high_to_mid_real_model_error: 236
- low_to_high_real_model_error: 172
- low_to_mid_real_model_error: 39
- mid_to_high_real_model_error: 524
- mid_to_low_real_model_error: 55

## Rejected Sources

- EduBenchEvaluator: 364
- deepseek-r1: 493
- deepseek-v3: 456
- gpt-4o: 382
- qwen3_4b_r0a_direct: 526
- qwq-plus: 464

## Dataset Variants

- edubench_r6a_qwen_only_score_real_dpo_train: 526 pairs; main score-only training candidate.
- edubench_r6b_multi_judge_score_real_dpo_train: 1250 pairs; main score-only training candidate.
- edubench_r6c_low_high_score_real_dpo_train: 632 pairs; main score-only training candidate.
- edubench_r6d_multi_judge_structured_chosen_real_score_dpo_train: 1250 pairs; auxiliary review candidate.
- edubench_r6e_multi_judge_reason_chosen_real_score_dpo_train: 1250 pairs; auxiliary review candidate.

## Recommended Review Order

1. Review `r6b_multi_judge_score_real_dpo_train` as the main real-DPO candidate.
2. Review `r6c_low_high_score_real_dpo_train` if the next DPO run should focus only on risk/protection.
3. Treat structured/reason chosen variants as ablations, not as the cleanest real-DPO baseline.
