# Exp35A EduDART-Cal Qualification

- Reference: model-reviewed silver, not human expert gold.
- Calibration-development: Exp33 representative train, 120 rows.
- Fresh general qualification: 120 rows; low-tail repeated-sample stress: 76 rows.
- EduDART-Cal fresh MAE/QWK: 0.6734245836989856/0.12416851441241705.
- Qwen hard fresh MAE/QWK: 0.4666666666666667/0.5969626168224299.
- Rounded human fresh MAE/QWK: 0.6166666666666667/0.39312977099236635.
- Low-tail EduDART-Cal label2 recall: 0.0.
- Qualification gate passed: False.
- Gates: `{"dev_test_access_zero": true, "high_to_low_not_worse_than_qwen_plus_0p01": true, "low_tail_label2_recall_ge_0p10": false, "low_to_high_not_worse_than_qwen": false, "mae_better_than_rounded_human_by_0p01": false, "mae_not_worse_than_qwen": false, "qwk_not_worse_than_qwen_minus_0p01": false, "qwk_not_worse_than_rounded_human": false}`.
- Dev/test access: 0/0; student training: none.
