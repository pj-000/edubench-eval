# Exp27E Provider Bias and Conflict-Adjudication Analysis

Exp27E is an offline analysis. It does not call APIs, train models, or use GPU.

## Decision

- recommend_use_both_for_361: True
- recommended_primary_teacher_for_full_train: `deepseek`
- recommend_selective_second_teacher: True
- recommend_gpt55_or_human_adjudication: True
- proceed_to_361_after_adjudication: True

## Provider Bias vs Original Human Label

| provider | n | MAE | signed bias | exact | adjacent | low->high | high->low |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen | 80 | 0.8750 | 0.3000 | 0.4375 | 0.8125 | 8 | 3 |
| deepseek | 80 | 0.9125 | 0.4375 | 0.4375 | 0.7500 | 8 | 2 |

## Conflict Types

| conflict_type | count |
|---|---:|
| both_teachers_disagree_with_human | 9 |
| deepseek_lenient_qwen_strict | 8 |
| derived_risk_disagreement | 18 |
| failure_bucket_disagreement | 24 |
| high_control_hard_conflict | 5 |
| high_human_teacher_low | 4 |
| low_human_teacher_high | 12 |
| possible_original_label_noise | 26 |
| possible_teacher_strictness_difference | 35 |
| qwen_lenient_deepseek_strict | 5 |
| score_gap_ge2 | 13 |

## Consensus Policy Simulation

| policy | high_trust | low_weight | review_only | excluded | estimated_train | high_conflicts_left |
|---|---:|---:|---:|---:|---:|---:|
| qwen_only_primary | 34 | 29 | 17 | 0 | 63 | 2 |
| deepseek_only_primary | 35 | 24 | 19 | 2 | 59 | 2 |
| exact_or_adjacent_consensus_else_review | 27 | 23 | 28 | 2 | 50 | 0 |
| deepseek_primary_qwen_selective_review | 29 | 12 | 37 | 2 | 41 | 0 |
| qwen_primary_deepseek_selective_review | 29 | 12 | 39 | 0 | 41 | 0 |
| original_human_with_teacher_quality_weight | 27 | 14 | 37 | 2 | 41 | 0 |

## Adjudication Queue

- full_queue_size: 43
- top40_size: 40
- top40_priority_counts: {1: 12, 2: 4, 4: 4, 5: 11, 6: 1, 7: 8}

The queue contains lightweight previews in CSV and train-only structured packets for later GPT5.5Pro
or human adjudication.

## Guardrails

- no training
- no API calls
- no GPU
- no dev labels are used
- no test labels are read
- dev/test samples are excluded from adjudication packets
- raw API outputs and full parsed teacher text remain local/ignored
