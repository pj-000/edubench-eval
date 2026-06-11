# Exp8 EduRisk Sanity Checklist

Status: checklist only. No training is run by this task.

## Planning-stage Checks

- [x] No training code in the Exp8 plan files.
- [x] No model training launched.
- [x] No API calls required.
- [x] No synthetic data proposed for the first run.
- [x] No Exp0-Exp7 result files modified.
- [x] First run uses question_seed42 human-only data.
- [x] First run has a fixed run id: `QD-ER1_EduRisk_human_only`.

## Pre-code Review Checks

- [ ] Confirm that the thesis framing needs a full-coverage training method, not only Exp7-C
      human-in-the-loop selective calibration.
- [ ] Confirm that the default hyperparameters are acceptable for the first run:
      `tau=0.7`, `alpha=0.3`, `beta_bce=0.5`, `lambda_LH=2.0`, `lambda_HL=0.5`,
      `class_balance_beta=0.99`, `normalized_cost=true`.
- [ ] Confirm that no synthetic data should be used in QD-ER1.
- [ ] Confirm that ablations remain blocked until the main run finishes and is reviewed.
- [ ] Confirm that primary decode is cumulative and secondary diagnostic decode is `argmax_q`.

## Future Implementation Checks

When code stage begins, add checks for:

- [ ] `q_raw_1..q_raw_5` are finite for every batch.
- [ ] `min(q_raw) >= -1e-6` before numerical guard.
- [ ] `abs(sum(q_raw) - 1) < tolerance` before numerical guard.
- [ ] `q_safe` is created only after q_raw sanity checks.
- [ ] `q_safe_k >= eps` after numerical guard.
- [ ] `sum_k q_safe_k = 1` within tolerance.
- [ ] `p_1 >= p_2 >= p_3 >= p_4` for the CORAL-style head.
- [ ] `L_softCE`, `L_risk`, `L_cumBCE`, and `L_EduRisk` are finite.
- [ ] class-balanced weights are finite and reported.
- [ ] `max(w_y) / min(w_y)` is reported before full training.
- [ ] all EduRisk parameters are loaded from config, not hard-coded in the loss.
- [ ] loss components are logged separately on train/dev.
- [ ] required logs include `L_total`, `L_softCE`, `L_risk`, `L_cumBCE`, `mean_weight`,
      `min_weight`, `max_weight`, `weighted_L_softCE`, `weighted_L_risk`, and
      `weighted_L_cumBCE`.
- [ ] no checkpoint or weight files are committed.

## Future Smoke-test Checks

Before a real run, use a tiny smoke configuration to verify:

- [ ] forward pass completes.
- [ ] backward pass completes.
- [ ] gradients are finite.
- [ ] no monotonic violation appears in decoded cumulative probabilities.
- [ ] output arrays/predictions have the same schema as QD-R1 where possible.
- [ ] output predictions include `pred_label_cum`, `pred_label_argmax`, and `pred_score_expected`.
- [ ] expected risk metrics can be computed.

The smoke test should not be treated as a scientific result.

## Future Full-run Checks

Before launching `QD-ER1_EduRisk_human_only`:

- [ ] Verify the dataset path points to question_seed42 human-only data.
- [ ] Verify synthetic paths are disabled.
- [ ] Verify output directory is Exp8-specific.
- [ ] Verify run id is exactly `QD-ER1_EduRisk_human_only`.
- [ ] Verify test set is not used for hyperparameter selection.
- [ ] Verify evaluation compares against QD-B0, QD-B1, and QD-R1.

## Future Evaluation Checks

Required final metrics:

- [ ] low_to_high_rate
- [ ] MAE
- [ ] QWK
- [ ] Kendall tau
- [ ] Acc@1
- [ ] Acc@2
- [ ] Acc@5
- [ ] low_signed_bias
- [ ] high_to_mid_or_low_rate
- [ ] monotonic_violation_rate
- [ ] ExpectedEduRisk overall
- [ ] ExpectedEduRisk by label group
- [ ] cumulative-vs-argmax decoding comparison
- [ ] loss component scale summary

## Failure-mode Review

If QD-ER1 fails, classify the failure:

- [ ] low_to_high improves but Acc@5 collapses versus QD-B1: reduce `lambda_LH` or increase
      `lambda_HL`.
- [ ] MAE/QWK worsen broadly: reduce `alpha` or increase cumulative BCE stability.
- [ ] low_to_high does not improve: inspect whether risk mass moves from labels 4/5 to 3 for true
      low-score samples.
- [ ] class weights dominate: keep `class_balance_beta=0.99` first; only later consider clipping
      with `weight_clip_min=0.5` and `weight_clip_max=3.0`.
- [ ] soft targets blur predictions: lower `tau` or test no-soft-target ablation.

## Code-stage Gate

Code implementation can start after this plan is reviewed and accepted. Until then, Exp8 remains a
design-only proposal.
