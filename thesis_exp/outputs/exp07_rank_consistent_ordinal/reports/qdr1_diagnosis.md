# QD-R1 CORAL Diagnosis

This diagnosis reads the completed QD-R1 raw predictions, arrays, run metadata, and summary tables.
It does not train a model, call an API, generate synthetic data, or modify raw predictions/arrays.

## Required Answers

- Did CORAL fix monotonicity? **YES.** Test monotonic_violation_rate is 0.0000.
- Did CORAL reduce low_to_high? **NO.** It matches QD-B0 and is worse than QD-B1.
- Did CORAL improve overall scoring? **NO.** Accuracy, MAE_label, QWK, Kendall, and Spearman are worse than QD-B0.
- Did CORAL hurt Acc@5? **NO.** Acc@5 increased, but this comes with stronger overestimation bias.
- Should Exp7-B start? **NOT_RECOMMENDED_YET.** First diagnose or change the score calibration/objective.

## Test Summary

| run | Accuracy | MAE_label | QWK | Kendall | low_to_high | Acc@5 | signed_bias_label | monotonic_violation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QD-B0_human_only_ordinary_ordinal | 0.6936 | 0.4019 | 0.5976 | 0.5293 | 0.5161 | 0.7873 | 0.1070 | 0.1985 |
| QD-B1_human_only_L1_weighted_ordinal | 0.6709 | 0.4279 | 0.6012 | 0.5266 | 0.4516 | 0.7419 | 0.0181 | 0.3119 |
| QD-R1_CORAL_human_only | 0.6555 | 0.4467 | 0.5272 | 0.4556 | 0.5161 | 0.8977 | 0.2829 | 0.0000 |

## Why Acc@5 Can Increase While MAE/QWK Worsen

QD-R1 shifts predictions upward. That helps many true label-5 examples stay at 5, improving Acc@5,
but the same upward shift overestimates lower and mid labels. This raises signed label bias, harms
exact
match on non-5 labels, increases severe errors, and weakens rank-sensitive metrics such as QWK and
Kendall.

## Low-score Errors

- Test low_to_high rate: 0.5161.
- Test low_to_mid rate: 0.2258.
- Test low exact rate: 0.2581.
- Low-to-high delta vs QD-B0: 0.0000.
- Low-to-high delta vs QD-B1: 0.0645.
- Test low pred distribution: pred 1: 8 (0.258), pred 2: 0 (0.000), pred 3: 7 (0.226), pred 4: 9 (0.290), pred 5: 7 (0.226).

## Overestimation

QD-R1 is overestimating scores. Its test signed label bias is 0.2829, compared with QD-B0 at 0.1070.
The mean predicted label by true-label table shows the upward drift is
especially damaging for low and mid labels.

## Threshold Diagnostics

Threshold/latent score status: **unavailable**.
Reason: learned threshold parameters and latent score arrays are not stored in the run output; only
logits/probabilities were exported.
The exported logits still allow checking rank consistency and approximate logit gap constancy, but
not
absolute learned tau values or mean latent score by true label.

## Recommendation

Do not start Exp7-B yet. CORAL solved the structural rank-consistency problem, but the main failure
mode
now appears to be upward score calibration/decision bias rather than independent-threshold
monotonicity.
A better next step is to diagnose threshold calibration or selection criteria before adding a new
Exp7-B variant.
