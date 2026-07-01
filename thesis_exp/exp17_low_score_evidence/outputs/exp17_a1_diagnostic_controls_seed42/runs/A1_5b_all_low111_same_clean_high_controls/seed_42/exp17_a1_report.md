# Exp17-A1 Evidence Head Run Report

- config_name: `A1_5b_all_low111_same_clean_high_controls`
- seed: 42
- beta: 0.1
- neg_ratio: 1:4
- positive_mode: `all_low_same_clean_high_controls`
- freeze_base: `False`
- train_evidence_head_only: `False`
- trainable_base_params: 595783692
- trainable_evidence_head_params: 1025
- init_checkpoint: `thesis_exp/outputs/exp16_boundary_linking/runs/qmr/seed_42/checkpoint_best/state_dict.pt`
- init_status: `loaded missing_base_keys=0`
- positive_count: 111
- negative_count: 444
- signal_count: 555
- signal_fraction: 0.16686710763680096
- signal_train_batch_count: 139
- hfe_training_mode: `full_ordinal_plus_signal_balanced_hfe`
- checkpoint_selection: `dev_mae`

## Dev Metrics

- MAE: 0.3920505871725384
- QWK: 0.5709011673854631
- low_to_high: 29 (0.5087719298245614)
- label2_recall: 0.0
- monotonic_violation_rate: 0.0

## Evidence Metrics

- h_auc_d1_hidden_vs_controls: 0.5122863247863247
- mean_h_d1_hidden: 0.16645695612980768
- mean_h_d1_controls: 0.1637963189019097
- evidence_delta_hidden_minus_control: 0.0026606372278979717
- mean_h_d1_evidence_positive: 0.14849853515625
- mean_h_d1_pairwise_low: 0.17429315476190477
- mean_h_d1_format_auxiliary: 0.07373046875
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