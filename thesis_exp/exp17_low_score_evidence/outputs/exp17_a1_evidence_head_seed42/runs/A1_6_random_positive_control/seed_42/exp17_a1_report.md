# Exp17-A1 Evidence Head Run Report

- config_name: `A1_6_random_positive_control`
- seed: 42
- beta: 0.1
- neg_ratio: 1:4
- positive_mode: `random_positive_control`
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

- MAE: 0.4010840108401084
- QWK: 0.569366024985551
- low_to_high: 28 (0.49122807017543857)
- label2_recall: 0.0
- monotonic_violation_rate: 0.0

## Evidence Metrics

- h_auc_d1_hidden_vs_controls: 0.3733974358974359
- mean_h_d1_hidden: 0.15147986778846154
- mean_h_d1_controls: 0.1792636447482639
- evidence_delta_hidden_minus_control: -0.02778377695980236
- mean_h_d1_evidence_positive: 0.1766357421875
- mean_h_d1_pairwise_low: 0.14839099702380953
- mean_h_d1_format_auxiliary: 0.11572265625
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