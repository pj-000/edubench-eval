# Exp54 SORC-DPO dev results

| Arm | Exact | MAE | Kendall | L2H | Recall-2 | Recall-5 | Forced close |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1_FIELD_DPO | 0.7043±0.0074 | 0.3404 | 0.5916 | 10.33 | 0.0238 | 0.8415 | 17.22% |
| P2_SORC_SCORE | 0.7033±0.0040 | 0.3424 | 0.5882 | 10.33 | 0.0238 | 0.8396 | 16.77% |
| P3_JOINT_SORC | 0.7053±0.0070 | 0.3394 | 0.5928 | 10.33 | 0.0238 | 0.8425 | 16.82% |
| P1_SYN_SEED42 | 0.7123 (seed42) | 0.3358 | 0.5994 | 10.00 | 0.0000 | 0.8609 | 24.10% |

Dev accessed: yes. Test accessed: no.
