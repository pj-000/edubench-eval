# Exp19-R5F2 Rejection Mining QC Report

R5F2 keeps actual train-side generated score-risk rejected outputs and reports whether there is
enough data for DPO scout.

- score-risk actual rejected pairs: 555
- unique low samples with score-risk rejected: 64
- evidence inconsistency pairs: 0
- high protection hard pairs in main: 555
- min ready pairs: 300
- min ready samples: 50
- ready_for_main_scout: `True`
- ready_for_real_only_scout: `True`

## Dataset Versions

- `edubench_r5f2_score_risk_main_dpo_train`: score-risk low rejected plus high-protection pairs.
- `edubench_r5f2_score_risk_plus_evidence_dpo_train`: main pairs plus low failure-evidence inconsistency pairs.
- `edubench_r5f2_real_only_small_dpo_train`: actual low-to-high generated rejected only.

No test split is read. Dev D1 annotations are not used for training.
