# QD-PR2 Setup Sanity

Status: `BLOCKED_MISSING_QDB1_CHECKPOINT`

| check | status | details |
| --- | --- | --- |
| QD-B1 checkpoint exists | BLOCKED | BLOCKED_MISSING_QDB1_CHECKPOINT: thesis_exp/artifacts/exp06_question_disjoint_baselines/checkpoints/QD-B1_human_only_L1_weighted_ordinal/best |
| train rows are human only | PASS |  |
| train has no synthetic rows | PASS |  |
| dev rows are human only | PASS |  |
| dev has no synthetic rows | PASS |  |
| test rows are human only | PASS |  |
| test has no synthetic rows | PASS |  |
| pair win_label > lose_label | PASS |  |
| high-comparability pair rates reported | PASS |  |
| pair weights finite | PASS |  |
| pair margins finite | PASS |  |
| anchor targets available | BLOCKED | on-the-fly QD-B1 reference logits from train pairs |
| L_point finite | PASS | 0.3274358490179159 |
| L_pair finite | PASS | 0.9972231124463287 |
| L_anchor finite | PASS | 0.48255269759823716 |
| L_mono finite | PASS | 0.0 |
| L_total finite | PASS | 0.6185733534393509 |
| toy monotonic regularizer positive only on violation | PASS | non_violation=0.0; violation=0.16313210846867754 |
| py_compile pass | PASS |  |
| bash -n pass | PASS |  |
| no checkpoint/weights tracked | PASS |  |
