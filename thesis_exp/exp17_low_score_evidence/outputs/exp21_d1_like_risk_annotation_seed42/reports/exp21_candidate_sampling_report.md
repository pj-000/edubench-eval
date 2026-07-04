# Exp21 D1-like Risk Annotation Candidate Sampling

Exp21 constructs annotation candidates only. It does not train, does not read test,
and does not use human rationale as model input.

## Exp20C Failure Modes Observed

- Automatic downgrade reduced aggregate low-to-high but demoted many true high-score cases.
- D1 hidden residual high-score predictions remain high, so the gate lacks enough precision/recall.
- The candidate package therefore includes both likely high-risk low cases and high-score false positives.

## Dev Audit Package

- downgraded_gold_high_to_3: 35
- rescued_d1_low_to_high: 5
- rescued_low_to_high: 1
- residual_d1_pred_ge4: 16
- residual_low_to_high: 1
- unchanged_safe_high_control: 40

## Train Prediction Availability

- train_predictions_available: False
- Train score/risk predictions are missing, so Exp21 generated LLaMA-Factory prediction assets.
- prediction_dataset: `thesis_exp/exp17_low_score_evidence/outputs/exp21_d1_like_risk_annotation_seed42/train_prediction_data/edubench_exp21_train_score_eval.json`
- score_config: `thesis_exp/exp17_low_score_evidence/outputs/exp21_d1_like_risk_annotation_seed42/train_predict_configs/r5g_a3_real_only_s50_b0p05_lr5em6.yaml`
- risk_config: `thesis_exp/exp17_low_score_evidence/outputs/exp21_d1_like_risk_annotation_seed42/train_predict_configs/r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6.yaml`
- Run `./thesis_exp/scripts/run_exp21_train_predictions.sh`, then rerun this Exp21 constructor.

## Train Annotation Candidates

- No train candidates sampled yet because train predictions are missing.

## Dominant Metrics / Languages

- Not available until train prediction mining is complete.

## Decision

- dev_audit_ready: True
- train_annotation_ready: False
- need_train_prediction_generation: True
- recommended_manual_annotation_count: 0
- next_step: `run_train_predictions`

## Guardrails

- Test split is not read.
- No model training is performed.
- Dev cases are diagnostic only and must not be used as training labels.
- Full train prediction prompts/configs are generated under ignored output folders.
