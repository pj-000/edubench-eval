# Exp17-A1 Evidence Head Run Report

- config_name: `A1_1b_a0_weak_random_high_negatives`
- seed: 42
- beta: 0.1
- neg_ratio: 1:4
- positive_mode: `a0_weak_random_high_negatives`
- freeze_base: `False`
- train_evidence_head_only: `False`
- trainable_base_params: 595783692
- trainable_evidence_head_params: 1025
- init_checkpoint: `thesis_exp/outputs/exp16_boundary_linking/runs/qmr/seed_42/checkpoint_best/state_dict.pt`
- init_status: `loaded missing_base_keys=0`
- positive_count: 76
- negative_count: 304
- signal_count: 380
- signal_fraction: 0.11425135297654841
- signal_train_batch_count: 95
- hfe_training_mode: `full_ordinal_plus_signal_balanced_hfe`
- checkpoint_selection: `dev_mae`

## Dev Metrics

- MAE: 0.4019873532068654
- QWK: 0.5437011957461575
- low_to_high: 31 (0.543859649122807)
- label2_recall: 0.0
- monotonic_violation_rate: 0.0

## Evidence Metrics

- h_auc_d1_hidden_vs_controls: 0.5592948717948718
- mean_h_d1_hidden: 0.10175499549278846
- mean_h_d1_controls: 0.0934600830078125
- evidence_delta_hidden_minus_control: 0.008294912484975964
- mean_h_d1_evidence_positive: 0.06072998046875
- mean_h_d1_pairwise_low: 0.11308361235119048
- mean_h_d1_format_auxiliary: 0.0279541015625
- dominant_question_group_id: `14ba3cb00f998348fe1c491eab066379d3bf192b`
- dominant_question_group_size: 20

## Guardrails

- No test split is read.
- Dev D1 annotation is used only for dev evidence evaluation, never as train labels.
- Human rationale text is not used as model input.
- Joint configs fine-tune the base model with an auxiliary evidence objective.
- Frozen-probe configs train only the evidence head.
- Frozen-probe configs use latest checkpoint selection because h does not affect dev MAE/QWK.
- The evidence head does not modify the ordinal scoring path.