# Exp7-C4 Selective Risk Calibration Robustness Report

Formal status: completed.

Training run: NO.

API calls: NO.

Synthetic data generation: NO.

Raw predictions/arrays modified: NO.

Test-set role: evaluation only. The C3 dev-selected policy is kept fixed.

## Recent Literature Motivation

- LLM abstention and selective prediction: recent abstention work frames refusal/review as a reliability mechanism for high-risk deployment, so C4 evaluates whether rejection removes risky grading cases rather than arbitrary cases.
- LLM confidence estimation and calibration: confidence and uncertainty estimates are central to selective prediction; C4 therefore tests an entropy-based confidence risk against random rejection.
- LLM conformal uncertainty and filtering: recent conformal uncertainty work motivates calibrated filtering of unreliable outputs; C4 is conformal-inspired but empirical only.
- LLM-as-a-Judge uncertainty: recent rating-based judge work explicitly studies uncertainty for discrete scores, matching this ordinal scorer setting.
- Ordinal risk control: conformal risk-control work for ordinal classification motivates targeting ordered-label risk, here low-score-to-high-score overestimation.
- SelectiveNet 2019 is foundational background for selective prediction, but this report is grounded primarily in newer LLM abstention, confidence, conformal uncertainty, judge uncertainty, and ordinal risk-control work.

## References

- LLM abstention / selective prediction: [Wen et al. 2024, Know Your Limits: A Survey of Abstention in Large Language Models](https://arxiv.org/abs/2407.18418).
- LLM confidence estimation and calibration: [Geng et al. 2024, A Survey of Confidence Estimation and Calibration in Large Language Models](https://aclanthology.org/2024.naacl-long.366/).
- LLM conformal uncertainty / filtering: [Wang et al. 2024, ConU: Conformal Uncertainty in Large Language Models with Correctness Coverage Guarantees](https://aclanthology.org/2024.findings-emnlp.404/).
- LLM-as-a-Judge uncertainty: [Sheng et al. 2025, Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction](https://aclanthology.org/2025.emnlp-main.569/).
- Conformal risk control for ordinal classification: [Xu et al. 2023, Conformal Risk Control for Ordinal Classification](https://proceedings.mlr.press/v216/xu23a.html).

## Review-budget Curve

| target review | actual review | coverage | low_to_high | MAE | QWK | Acc@5 | capture |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.0000 | 1.0000 | 0.4516 | 0.4279 | 0.6012 | 0.7419 | 0.0000 |
| 0.05 | 0.0499 | 0.9501 | 0.4138 | 0.4119 | 0.6224 | 0.7568 | 0.1429 |
| 0.10 | 0.0997 | 0.9003 | 0.4138 | 0.4082 | 0.6276 | 0.7683 | 0.1429 |
| 0.15 | 0.1496 | 0.8504 | 0.4138 | 0.3941 | 0.6454 | 0.7888 | 0.1429 |
| 0.20 | 0.1995 | 0.8005 | 0.3200 | 0.3783 | 0.6649 | 0.8125 | 0.4286 |
| 0.25 | 0.2493 | 0.7507 | 0.3333 | 0.3688 | 0.6795 | 0.8320 | 0.4286 |

## Selective vs Random Rejection at 20%

- low_to_high_non_rejected: selective 0.3200 vs random mean 0.4503; gap -0.1303.
- MAE_non_rejected: selective 0.3783 vs random mean 0.4286; gap -0.0504.
- QWK_non_rejected: selective 0.6649 vs random mean 0.6020; gap 0.0629.
- Acc@5_non_rejected: selective 0.8125 vs random mean 0.7425; gap 0.0700.

## Selective vs Oracle Upper Bound at 20%

- Oracle low_to_high_non_rejected: 0.0000; selective remains 0.3200.
- Oracle captures 1.0000 of low_to_high errors; selective captures 0.4286.
- Oracle rejection is not deployable because it uses test labels; it is only a headroom estimate.

## Rejected Sample Distribution at 20%

- Rejected samples: 220 of 1103.
- Low-to-high captured: 6/14.
- Low-to-high enrichment among rejected samples: 2.1487.
- True-low enrichment among rejected samples: 0.9704.
- Pred-high enrichment among rejected samples: 1.0208.

## Risk-control Framing

- Target risk: low_to_high among auto-scored non-rejected samples.
- Target coverage: around 0.8000; achieved 0.8005.
- Achieved low_to_high: 0.3200; target <= 0.3500.
- Random baseline at same budget: 0.4503.
- Oracle upper bound at same budget: 0.0000.
- Target achieved: YES.
- Guarantee claim: Empirical risk-control analysis; no finite-sample conformal guarantee claimed.

## Thesis Recommendation

QD-B1 raw + entropy_confidence_risk at a 20% review budget is suitable as the thesis final method if
framed as human-in-the-loop selective scoring. It should not be framed as a full-coverage automatic
scorer.

## Limitations

- Non-rejected metrics exclude samples sent to review.
- The analysis is empirical risk control, not a formal conformal finite-sample guarantee.
- The final method assumes a review workflow can absorb roughly 20% of cases.

## Suggested Paper Wording

We use a dev-selected entropy-confidence rejection rule to route high-risk ordinal scoring cases to
human review. On the held-out test set, at 80.1% automatic coverage, the method reduces
low-score-to-high-score overestimation from 45.2% to 32.0%, outperforming a 100-seed random
rejection baseline and remaining below the 35% target risk. We report this as empirical risk control
for human-in-the-loop scoring, without claiming a finite-sample conformal guarantee.
