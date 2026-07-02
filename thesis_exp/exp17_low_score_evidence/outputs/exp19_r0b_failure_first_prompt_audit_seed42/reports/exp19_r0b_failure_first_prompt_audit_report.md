# Exp19-R0B Failure-First Prompt Audit

## Setup

- Dataset: `thesis_exp/data/splits/question_seed42/dev.jsonl`.
- R0A baseline dir: `thesis_exp/exp17_low_score_evidence/outputs/exp19_r0a_qwen4b_direct_baseline_seed42`.
- D1 evaluation dir: `thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/summary_human_rationale_recovered`.
- Model: `/home/jpang/models/modelscope/Qwen/Qwen3-4B`.
- Backend used: `vllm`.
- Training: none.
- Test split: not read by default.
- Prompt inputs: question, answer, metric, rubric, and metadata only.
- Human score, gold label, human rationale, D1 labels, failure-type gold, existing judge scores, and Exp16A predictions are not included in prompts.

## Prompt Metrics

| prompt | n | parse | MAE | QWK | low-to-high | label2 recall | D1 hidden pred>=4 | failure micro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `R0B_0_direct_reproduction` | 1107 | 1.0000 | 1.7742 | 0.0402 | 0.2807 | 0.8158 | 0.2308 | 0.1304 |
| `R0B_1_failure_first` | 1107 | 1.0000 | 1.7073 | 0.0244 | 0.2281 | 0.8947 | 0.0385 | 0.1569 |
| `R0B_2_rubric_clause_first` | 1107 | 1.0000 | 1.9386 | 0.0051 | 0.1930 | 0.9737 | 0.0000 | 0.1154 |
| `R0B_3_conservative_low_score_cap` | 1107 | 1.0000 | 1.6459 | 0.0312 | 0.1579 | 0.8684 | 0.0769 | 0.0800 |

## Answers

- Does failure-first prompting reduce low-to-high vs R0A direct dev baseline? Best prompt `R0B_3_conservative_low_score_cap` has low-to-high `0.1579`; R0A dev is `0.8246`; absolute improvement is `0.6667`.
- Which prompt config is best? `R0B_3_conservative_low_score_cap`.
- Does rubric-clause-first reduce D1 hidden pred>=4? `R0B_2_rubric_clause_first` D1 pred>=4 is `0.0000`.
- Does conservative score-cap improve label2 recall? `R0B_3_conservative_low_score_cap` label2 recall is `0.8684`.
- Does parse success remain acceptable? Best prompt parse success is `1.0000`.
- Does model still over-score low samples? Best prompt low-to-high is `0.1579`.
- Should lock prompt and run test/all_5536? `True`.
- Should proceed to LoRA/SFT? `False`.

## Decision

```json
{
  "best_prompt_config": "R0B_3_conservative_low_score_cap",
  "d1_hidden_pred_ge4_best": 0.07692307692307693,
  "dev_label2_recall_best": 0.868421052631579,
  "dev_low_to_high_best": 0.15789473684210525,
  "lock_prompt_for_test_all": true,
  "parse_success_best": 1.0,
  "proceed_to_lora": false,
  "prompt_audit_success": true,
  "reason": "Prompt audit passes all gates; lock this prompt for held-out evaluation.",
  "recommended_lora_variant": ""
}
```

## Guardrails

- Full internal predictions are written under `predictions/full_predictions_internal.jsonl` and must not be committed.
- Raw prompts and raw answers are not written to lightweight report artifacts.
