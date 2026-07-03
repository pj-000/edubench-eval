# Exp19-R5C Generation Mining Plan

R5C score-risk pairs should be strengthened with additional train-only candidate generations.

## Needed Risk Types

- low_to_high_score_risk
- high_to_low_score_risk
- mid_score_calibration

## Subsets

- train low reasoned plus all train y<=2
- train clean high controls plus stratified train y>=4
- train y=3

## Adapters

- r1b_score_only_balanced
- r2n_reason_score_natural
- r2c_clean_reason_score_balanced
- r4b_shuffled_reason_balanced

## Generation

- temperature: 0.7
- top_p: 0.9
- max_new_tokens: 256
- num_samples_per_input: 4

## Filters

- low: score >= 4, or score == 3 with no_major_failure and score_cap null
- high: score <= 2, or score == 3 with false failure and score_cap <= 3
- mid: score <= 1 or score >= 5

Do not read test or use dev/D1 annotations for training.
