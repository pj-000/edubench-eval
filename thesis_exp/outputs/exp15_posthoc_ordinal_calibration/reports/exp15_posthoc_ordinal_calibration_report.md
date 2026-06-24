# Exp15 Post-hoc Ordinal Calibration

Status: `COMPLETED`
Recommendation: `NO_FORMAL_RECOMMENDED`

Exp15 is a dev-only post-hoc calibration diagnosis. It does not train a model and does not use
test metrics for calibrator selection, checkpoint selection, or tuning.

The selection rule is `pava_mae_guard_low_to_high_then_label2_then_calibration`: keep
calibrators inside the PAVA baseline dev MAE guard, then prefer lower dev low-to-high
count, lower label-2 low-to-high count, lower high-to-low count, and better calibration.

## Selected Dev Calibrators

| source | calibrator | dev MAE | dev low-to-high | delta vs pava | dev label2 low-to-high | dev brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| exp09_qdpr2:formal:seed42 | temp3p0_pava_t0p50 | 0.4851 | 28/57 | 0 | 26/38 | 0.0790 |
| exp13:score_proj_l2h_lam0p20:seed42 | temp3p0_pava_t0p50 | 0.4827 | 28/57 | 0 | 26/38 | 0.0792 |
| exp13:score_proj_l2h_lam0p20:seed43 | temp3p0_pava_t0p50 | 0.4857 | 28/57 | 0 | 26/38 | 0.0787 |
| exp13:score_proj_l2h_lam0p20:seed44 | temp3p0_pava_t0p50 | 0.4857 | 28/57 | 0 | 26/38 | 0.0793 |

## Selected Test Diagnostics

| source | calibrator | test MAE | test low-to-high | test label2 low-to-high | test QWK |
| --- | --- | ---: | ---: | ---: | ---: |
| exp09_qdpr2:formal:seed42 | temp3p0_pava_t0p50 | 0.4201 | 12/31 | 11/22 | 0.6057 |
| exp13:score_proj_l2h_lam0p20:seed42 | temp3p0_pava_t0p50 | 0.4173 | 13/31 | 12/22 | 0.6129 |
| exp13:score_proj_l2h_lam0p20:seed43 | temp3p0_pava_t0p50 | 0.4107 | 14/31 | 13/22 | 0.6134 |
| exp13:score_proj_l2h_lam0p20:seed44 | temp3p0_pava_t0p50 | 0.4234 | 14/31 | 13/22 | 0.5965 |

## Dev Risk-Accuracy Tradeoff

Some calibrators lower dev low-to-high but fall outside the PAVA baseline MAE guard.

| source | calibrator | dev MAE | dev low-to-high | delta vs pava | MAE delta vs pava |
| --- | --- | ---: | ---: | ---: | ---: |
| exp13:score_proj_l2h_lam0p20:seed42 | pava_q3_tau0p98 | 0.6079 | 20/57 | -8 | 0.1253 |
| exp13:score_proj_l2h_lam0p20:seed43 | pava_q3_tau0p98 | 0.6001 | 21/57 | -7 | 0.1144 |
| exp13:score_proj_l2h_lam0p20:seed44 | pava_q3_tau0p98 | 0.6031 | 21/57 | -7 | 0.1174 |
| exp09_qdpr2:formal:seed42 | pava_q3_tau0p98 | 0.6092 | 21/57 | -7 | 0.1241 |
| exp13:score_proj_l2h_lam0p20:seed42 | pava_q3_tau0p95 | 0.5589 | 22/57 | -6 | 0.0762 |

## Method Notes

- Raw, PAVA, temperature, threshold, q3-bias, and threshold-wise isotonic calibrators are compared.
- Isotonic models are fitted on dev threshold labels only and then applied unchanged to test.
- Prediction arrays, logits, checkpoints, and raw JSONL predictions are not written to Exp15 outputs.
