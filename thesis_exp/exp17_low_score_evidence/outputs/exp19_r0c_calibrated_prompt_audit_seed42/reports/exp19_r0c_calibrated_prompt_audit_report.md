# Exp19-R0C Calibrated Failure-First Prompt Audit

## Setup

- Dataset: `thesis_exp/data/splits/question_seed42/dev.jsonl`.
- R0A baseline dir: `thesis_exp/exp17_low_score_evidence/outputs/exp19_r0a_qwen4b_direct_baseline_seed42`.
- R0B baseline dir: `thesis_exp/exp17_low_score_evidence/outputs/exp19_r0b_failure_first_prompt_audit_seed42`.
- D1 evaluation dir: `thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/summary_human_rationale_recovered`.
- Model: `/home/jpang/models/modelscope/Qwen/Qwen3-4B`.
- Backend used: `vllm`.
- Training: none.
- Test split: not read by default.
- Prompt inputs: question, answer, metric, rubric, and metadata only.
- Human score, gold label, human rationale, D1 labels, failure-type gold, existing judge scores, and Exp16A predictions are not included in prompts.

## Prompt Metrics

| prompt | n | parse | MAE | QWK | bias | low-to-high | high-to-low | label2 recall | label5 recall | D1 hidden pred>=4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `R0C_0_exact_r0a_reproduction` | 1107 | 1.0000 | 0.7633 | 0.0451 | 0.4399 | 0.8246 | 0.0360 | 0.1053 | 0.8921 | 0.7692 |
| `R0C_1_balanced_failure_first` | 1107 | 0.9973 | 1.7545 | 0.0034 | -1.4737 | 0.2632 | 0.6688 | 0.9474 | 0.2122 | 0.0385 |
| `R0C_2_evidence_required_cap` | 1107 | 0.9973 | 2.1803 | -0.0222 | -2.0643 | 0.2281 | 0.8741 | 0.8421 | 0.0216 | 0.1538 |
| `R0C_3_two_pass_balanced` | 1107 | 0.9973 | 1.9411 | 0.0020 | -1.7292 | 0.2456 | 0.7725 | 0.9211 | 0.1385 | 0.0385 |
| `R0C_4_high_score_protection` | 1107 | 0.9955 | 1.9728 | 0.0075 | -1.7550 | 0.1754 | 0.7873 | 0.8947 | 0.1439 | 0.0000 |

## Baseline References

- R0A dev: MAE `0.7651`, QWK `0.0433`, low-to-high `0.8246`, label2 recall `0.1053`.
- R0B best dev: MAE `1.6459`, QWK `0.0312`, low-to-high `0.1579`, label2 recall `0.8684`.

## Answers

- Does exact R0A reproduction match R0A dev metrics? MAE delta `-0.0018`, low-to-high delta `0.0000`, label2 recall delta `0.0000`.
- Which calibrated prompt best balances low-to-high and MAE/QWK? `R0C_2_evidence_required_cap`.
- Does the prompt avoid R0B over-conservatism? Best bias `-2.0643`, high-to-low `0.8741`, MAE `2.1803`.
- Does high_to_low remain acceptable? `0.8741`.
- Does label5 recall remain acceptable? `0.0216`.
- Does D1 hidden pred>=4 stay low? `0.1538`.
- Does failure type accuracy improve? Best failure micro-F1 `0.0784`.
- Should lock prompt and run held-out test/all_5536? `False`.
- Should proceed to LoRA/SFT? `True`.

## Decision

```json
{
  "best_prompt_config": "R0C_2_evidence_required_cap",
  "calibrated_prompt_success": false,
  "d1_hidden_pred_ge4_best": 0.15384615384615385,
  "dev_MAE_best": 2.1802536231884058,
  "dev_QWK_best": -0.022202015881943815,
  "dev_high_to_low_best": 0.8740740740740741,
  "dev_label2_recall_best": 0.8421052631578947,
  "dev_label5_recall_best": 0.02158273381294964,
  "dev_low_to_high_best": 0.22807017543859648,
  "dev_signed_bias_best": -2.0643115942028984,
  "lock_prompt_for_test_all": false,
  "parse_success_best": 0.997289972899729,
  "proceed_to_lora": true,
  "reason": "Prompt improves low-to-high meaningfully, but calibration or high-score protection remains weak.",
  "recommended_lora_variant": "score + failure evidence + score_cap LoRA/SFT"
}
```

## Guardrails

- Full internal predictions are written under `predictions/full_predictions_internal.jsonl` and must not be committed.
- Raw prompts and raw answers are not written to lightweight report artifacts.
