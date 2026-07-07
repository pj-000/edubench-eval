# Exp19-R7 Reason-Recovered Real DPO Dataset Report

Exp19-R7 rebuilds the DPO data around original human rationales. It is train-only data for review
before DPO training.

## Construction

- input prompt: question + answer + metric + rubric + metadata.
- chosen response: train gold score, optionally with recovered original human reason.
- rejected response: real wrong scalar score from Qwen3-4B R0A or existing judge scores.
- rejected responses are not synthetic templates.
- human reasons are never inserted into the user prompt.

## Leakage Guard

- train rows used for construction: `3326`
- dev rows read only for ID/question-key leakage guard: `1107`
- test rows read only for ID/question-key leakage guard: `1103`
- dev/test labels are not used.
- any sample/question overlap with dev/test makes the script fail.

| dataset | pairs | leakage_pass | human_reason_in_prompt_count |
|---|---:|---|---:|
| `edubench_r7a_score_only_reason_covered_real_dpo_train` | 749 | `True` | 0 |
| `edubench_r7b_human_reason_chosen_real_score_dpo_train` | 749 | `True` | 0 |
| `edubench_r7c_low_high_human_reason_chosen_real_score_dpo_train` | 429 | `True` | 0 |
| `edubench_r7d_strict_label_consistent_reason_real_dpo_train` | 429 | `True` | 0 |

## Reason Recovery

- pair pool before reason filtering: `1250`
- pair pool with recovered reason: `749`
- pair pool with label-consistent recovered reason: `749`
- missing reason files: `none`

## Dataset Variants

- `edubench_r7a_score_only_reason_covered_real_dpo_train`: 749 pairs; score-only control.
- `edubench_r7b_human_reason_chosen_real_score_dpo_train`: 749 pairs; reason-aware DPO candidate.
- `edubench_r7c_low_high_human_reason_chosen_real_score_dpo_train`: 429 pairs; reason-aware DPO candidate.
- `edubench_r7d_strict_label_consistent_reason_real_dpo_train`: 429 pairs; reason-aware DPO candidate.

## Recommended Review Order

1. Review `edubench_r7d_strict_label_consistent_reason_real_dpo_train` first. It is the cleanest reason-aware candidate.
2. Review `edubench_r7c_low_high_human_reason_chosen_real_score_dpo_train` if the goal is low/high risk control.
3. Keep `edubench_r7a_score_only_reason_covered_real_dpo_train` as a score-only control on the same reason-covered sample pool.

## Caveats

- Rejected responses are real scalar score outputs, not full rejected reasons.
- Chosen reasons are recovered from original human rationale files, but some are not label-consistent with the rounded final label.
- The strict R7D variant filters to label-consistent recovered rationales to reduce this risk.
