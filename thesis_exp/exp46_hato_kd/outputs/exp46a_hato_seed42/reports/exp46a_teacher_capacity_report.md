# Exp46A Teacher Capacity Gate

Decision: **TEACHER_CAPACITY_NO_GO**

The 4B teacher uses only the locked human score distribution and the same question-key GroupCV split; no teacher relabeling, dev, or test data are used.

## Gate checks

- label2_recall: FAIL
- label2_correct: FAIL
- label2_precision: FAIL
- low_to_high: FAIL
- overall_gain: FAIL
- exact_protection: FAIL
- label5_protection: PASS
- high_to_low_protection: PASS
- bias_protection: FAIL
- bootstrap_no_significant_overall_harm: PASS

## Key metrics

- Label2: 0/52 correct; recall=0.0000; precision=0.0000.
- Low-to-high: teacher=0.7895; 0.6B E4=0.7632.
- MAE/QWK/Exact/Kendall: 0.4043 / 0.4420 / 0.6590 / 0.4805.

No test data were read.
