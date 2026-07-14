# Exp41A RUBRIC-Bridge GroupCV report

- Status: **GROUPCV_STOP**
- V0H metrics: `{"Bin_Agreement": 0.8696307460437076, "Exact_Match": 0.6318764129615675, "Kendall_tau": 0.41726492150135824, "MAE": 0.4431047475508666, "QWK": 0.34120610991261957, "Signed_Bias": 0.1281085154483798, "abs_Signed_Bias": 0.1281085154483798, "expected_score_MAE": 0.4930197673364058, "high_n": 2327.0, "high_to_low_count": 4.0, "high_to_low_rate": 0.0017189514396218307, "label1_recall": 0.0, "label2_recall": 0.0, "label3_recall": 0.11155378486055777, "label4_recall": 0.6701902748414377, "label5_recall": 0.7349746560463433, "low_n": 76.0, "low_to_high_count": 71.0, "low_to_high_rate": 0.9342105263157895, "n": 2654.0, "pred_count_1": 1.0, "pred_count_2": 3.0, "pred_count_3": 76.0, "pred_count_4": 1195.0, "pred_count_5": 1379.0, "variant": "v0h_human_soft"}`
- V3 RUBRIC-Bridge metrics: `{"Bin_Agreement": 0.8636021100226073, "Exact_Match": 0.6443104747550866, "Kendall_tau": 0.4501825843704151, "MAE": 0.4284099472494348, "QWK": 0.36934518942394245, "Signed_Bias": 0.13602110022607386, "abs_Signed_Bias": 0.13602110022607386, "expected_score_MAE": 0.471411154805552, "high_n": 2327.0, "high_to_low_count": 2.0, "high_to_low_rate": 0.0008594757198109154, "label1_recall": 0.0, "label2_recall": 0.0, "label3_recall": 0.10358565737051793, "label4_recall": 0.6596194503171248, "label5_recall": 0.7675597393193339, "low_n": 76.0, "low_to_high_count": 67.0, "low_to_high_rate": 0.881578947368421, "n": 2654.0, "pred_count_1": 2.0, "pred_count_2": 2.0, "pred_count_3": 94.0, "pred_count_4": 1137.0, "pred_count_5": 1419.0, "variant": "v3_rubric_bridge"}`
- Gate checks: `{"absolute_bias_guard": true, "exact_guard": true, "high_to_low_guard": true, "kendall_guard": true, "label2_guard": true, "label5_guard": true, "low_to_high_guard": true, "no_significant_bootstrap_harm_mae_qwk_exact": true, "outperforms_v1_raw_rubric": false, "outperforms_v2_deterministic_checklist": false, "outperforms_v4_shuffled": true, "primary_gain_mae_or_qwk": true}`
- Recommend run multiseed: `false`
- Stop LLM rubric compiler route: `true`
- The Qwen teacher compiled rubrics without seeing answers or labels.
- Human labels were not replaced; training used standard hard/soft cross-entropy only.
- Held-out evaluation contains original human rows only with question-key-disjoint folds.
- No paper-like dev/test data were accessed.
