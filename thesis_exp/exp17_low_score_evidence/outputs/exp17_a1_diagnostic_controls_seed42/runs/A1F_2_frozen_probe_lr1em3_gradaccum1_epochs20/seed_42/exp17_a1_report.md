# Exp17-A1 Evidence Head Run Report

- config_name: `A1F_2_frozen_probe_lr1em3_gradaccum1_epochs20`
- seed: 42
- beta: 0.1
- neg_ratio: 1:4
- positive_mode: `a0_weak`
- freeze_base: `True`
- train_evidence_head_only: `True`
- trainable_base_params: 0
- trainable_evidence_head_params: 1025
- init_checkpoint: `thesis_exp/outputs/exp16_boundary_linking/runs/qmr/seed_42/checkpoint_best/state_dict.pt`
- init_status: `loaded missing_base_keys=0`
- positive_count: 76
- negative_count: 304
- signal_count: 380
- signal_fraction: 0.11425135297654841
- signal_train_batch_count: 95
- hfe_training_mode: `signal_only`
- checkpoint_selection: `latest`

## Dev Metrics

- MAE: 0.3992773261065944
- QWK: 0.5532253604287027
- low_to_high: 29 (0.5087719298245614)
- label2_recall: 0.0
- monotonic_violation_rate: 0.0

## Evidence Metrics

- h_auc_d1_hidden_vs_controls: 0.28739316239316237
- mean_h_d1_hidden: 0.007962520305926982
- mean_h_d1_controls: 0.012004375457763672
- evidence_delta_hidden_minus_control: -0.0040418551518366895
- mean_h_d1_evidence_positive: 0.006212949752807617
- mean_h_d1_pairwise_low: 0.008672169276646205
- mean_h_d1_format_auxiliary: 5.817413330078125e-05
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