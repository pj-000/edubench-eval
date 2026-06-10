# Exp6 Synthetic-Augmented Training Datasets

Overall dataset build status: **PASS**

This stage builds data only. It does not call an API, generate new synthetic rows, or train models.

## Dataset Summary

| dataset | split | rows | human | synthetic |
| --- | --- | ---: | ---: | ---: |
| QD-S0_human_only | train | 3326 | 3326 | 0 |
| QD-S0_human_only | dev | 1107 | 1107 | 0 |
| QD-S0_human_only | test | 1103 | 1103 | 0 |
| QD-S1_human_plus_synthetic_ordinal | train | 3710 | 3326 | 384 |
| QD-S1_human_plus_synthetic_ordinal | dev | 1107 | 1107 | 0 |
| QD-S1_human_plus_synthetic_ordinal | test | 1103 | 1103 | 0 |
| QD-S2_human_plus_synthetic_L1 | train | 3710 | 3326 | 384 |
| QD-S2_human_plus_synthetic_L1 | dev | 1107 | 1107 | 0 |
| QD-S2_human_plus_synthetic_L1 | test | 1103 | 1103 | 0 |
| QD-S3_synthetic_pretrain_then_human_finetune | stage1_train | 384 | 0 | 384 |
| QD-S3_synthetic_pretrain_then_human_finetune | stage2_train | 3326 | 3326 | 0 |
| QD-S3_synthetic_pretrain_then_human_finetune | train | 3326 | 3326 | 0 |
| QD-S3_synthetic_pretrain_then_human_finetune | dev | 1107 | 1107 | 0 |
| QD-S3_synthetic_pretrain_then_human_finetune | test | 1103 | 1103 | 0 |

## Synthetic Pool

- Synthetic rows used: 384
- Label distribution: 1=168, 2=168, 3=48
- Language distribution: en=194, zh=190
- Labels remain pseudo labels and must not be interpreted as human labels.

## Training Gate

- Can Exp6 model training start? YES
- Allowed training runs if the gate is YES: QD-S1 ordinary ordinal; QD-S2 L1 weighted ordinal; QD-S3 synthetic pretrain -> human fine-tune.
- Synthetic-only main training is not built because the synthetic pool only covers labels 1/2/3.
