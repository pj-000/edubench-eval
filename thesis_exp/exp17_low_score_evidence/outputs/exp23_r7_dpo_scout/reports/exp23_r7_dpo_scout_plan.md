# Exp23 R7 DPO Scout Plan

Exp23 trains ordinary DPO scouts to check whether recovered human rationale is useful before
introducing ORC-DPO or SRC-DPO formula changes.

## Runs

| run | family | pairs | purpose |
|---|---|---:|---|
| `r7d_reason_real_s100_b0p03_lr5em6` | `human_reason_real_error` | 429 | R7D chosen has recovered human rationale plus gold score; rejected is real wrong score. |
| `r7e_matched_score_only_s100_b0p03_lr5em6` | `matched_score_only_control` | 429 | R7E uses the exact R7D pair pool but removes chosen rationale. |
| `r7f_score_reason_consistency_s100_b0p03_lr5em6` | `score_reason_consistency_counterfactual` | 856 | R7F trains consistency between recovered human reason and final score. |

## Interpretation

- R7D vs R7E is the key fair comparison because they share the same source pair pool.
- R7F is not a natural real-error dataset; it is a score-reason consistency auxiliary scout.
- This step uses ordinary DPO only. It should not be described as the final algorithmic method.

## Training Defaults

- init adapter: `saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora`
- max_steps: 100
- pref_beta: 0.03
- pref_ftx: 0.05
- learning_rate: 5e-06
- per_device_train_batch_size: 1
- gradient_accumulation_steps: 8

## Guardrails

- Training datasets are train-only DPO data.
- This preparation script does not read dev/test labels.
- Existing R7/Exp22 leakage audits are copied only as prior guardrail evidence.
- Do not submit checkpoints, raw predictions, logs, full generated outputs, numpy arrays, or model weights.

## Prior Leakage Summary

| source | dataset | pairs | dev sample overlap | dev question overlap | test sample overlap | test question overlap | reason in prompt | pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `r7` | `r7a_score_only_reason_covered` | 749 | 0 | 0 | 0 | 0 | 0 | True |
| `r7` | `r7b_human_reason_chosen` | 749 | 0 | 0 | 0 | 0 | 0 | True |
| `r7` | `r7c_low_high_human_reason_chosen` | 429 | 0 | 0 | 0 | 0 | 0 | True |
| `r7` | `r7d_strict_label_consistent_reason` | 429 | 0 | 0 | 0 | 0 | 0 | True |
| `exp22` | `edubench_r7e_matched_score_only_strict_real_dpo_train` | 429 | 0 | 0 | 0 | 0 | 0 | True |
| `exp22` | `edubench_r7f_score_reason_consistency_dpo_train` | 856 | 0 | 0 | 0 | 0 | 0 | True |
