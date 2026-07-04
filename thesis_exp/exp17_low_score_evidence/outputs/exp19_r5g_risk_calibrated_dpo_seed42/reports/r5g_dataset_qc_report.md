# Exp19-R5G Risk-Calibrated DPO Data QC

R5G constructs a small DPO scout after R5F2 showed that real low-to-high rejected responses reduce
low-score overestimation but can over-correct and hurt high-score protection.

## Datasets

| dataset | n | low-to-high | high-to-low | actual rejected | hard synthetic | duplicate pair rate |
|---|---:|---:|---:|---:|---:|---:|
| `edubench_r5g_real_only_dpo_train` | 555 | 555 | 0 | 555 | 0 | 0.0000 |
| `edubench_r5g_ratio_70_30_dpo_train` | 793 | 555 | 238 | 555 | 238 | 0.0000 |
| `edubench_r5g_ratio_60_40_dpo_train` | 925 | 555 | 370 | 555 | 370 | 0.0000 |
| `edubench_r5g_ratio_50_50_dpo_train` | 1110 | 555 | 555 | 555 | 555 | 0.0000 |

## Scout Matrix

- Group A: lighter real-only DPO from R2c, using steps/beta/lr sweeps.
- Group B: ratio-calibrated low-risk/high-protection DPO from R2c at 70/30, 60/40, and 50/50.
- No test split is read.
- D1 annotations are not used for training labels.
- Full DPO JSON remains under gitignored `data/`.

## Source Summary

- source rows summarized: 279
