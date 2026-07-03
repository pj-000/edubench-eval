# Exp19 LLaMA-Factory Dataset Validation

- dataset_dir: `thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42`
- dataset_info: `thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42/dataset_info.json`
- datasets checked: 12
- recovered rationale strings checked: 80
- total user-prompt leakage count: 0
- errors: 0

## Datasets

- edubench_exp19_dev_score_eval: count=1107, ranking=False, leakage=0, labels={'1': 19, '2': 38, '3': 105, '4': 389, '5': 556}
- edubench_r1_score_only_balanced_train: count=3000, ranking=False, leakage=0, labels={'1': 445, '2': 455, '3': 525, '4': 619, '5': 956}
- edubench_r1_score_only_train: count=3326, ranking=False, leakage=0, labels={'1': 58, '2': 53, '3': 297, '4': 1163, '5': 1755}
- edubench_r2_clean_reason_score_balanced_train: count=3000, ranking=False, leakage=0, labels={'1': 520, '2': 380, '3': 525, '4': 621, '5': 954}
- edubench_r2_reason_score_balanced_train: count=3000, ranking=False, leakage=0, labels={'1': 445, '2': 455, '3': 525, '4': 619, '5': 956}
- edubench_r2_reason_score_train: count=3326, ranking=False, leakage=0, labels={'1': 58, '2': 53, '3': 297, '4': 1163, '5': 1755}
- edubench_r3_reason_rationale_balanced_train: count=3000, ranking=False, leakage=0, labels={'1': 465, '2': 435, '3': 525, '4': 643, '5': 932}
- edubench_r3_reason_rationale_train: count=3326, ranking=False, leakage=0, labels={'1': 58, '2': 53, '3': 297, '4': 1163, '5': 1755}
- edubench_r4_shuffled_reason_balanced_train: count=3000, ranking=False, leakage=0, labels={'1': 520, '2': 380, '3': 525, '4': 621, '5': 954}
- edubench_r4_shuffled_reason_control_train: count=3000, ranking=False, leakage=0, labels={'1': 445, '2': 455, '3': 525, '4': 619, '5': 956}
- edubench_r5_high_protection_dpo_control_train: count=6466, ranking=True, leakage=0, labels={}
- edubench_r5_risk_balanced_dpo_train: count=3000, ranking=True, leakage=0, labels={}
