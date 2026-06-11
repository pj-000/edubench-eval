# Exp8 EduRisk Experiment Matrix

Status: design only. No training is launched by this plan.

## Main Comparison

| run_id | role | data | input | head/objective | synthetic | purpose |
| --- | --- | --- | --- | --- | --- | --- |
| QD-B0_human_only_ordinary_ordinal | baseline | question_seed42 human-only | A4 | ordinary ordinal | no | clean ordinary ordinal baseline |
| QD-B1_human_only_L1_weighted_ordinal | baseline | question_seed42 human-only | A4 | L1-weighted ordinal | no | best Exp5-style weighted baseline |
| QD-R1_CORAL_human_only | baseline | question_seed42 human-only | A4 | CORAL rank-consistent | no | rank-consistent Exp7 baseline |
| QD-ER1_EduRisk_human_only | proposed | question_seed42 human-only | A4 | EduRisk Ordinal Loss | no | first loss-innovation run |

## First Run Configuration

| field | value |
| --- | --- |
| run_id | QD-ER1_EduRisk_human_only |
| data split | question-disjoint seed42 |
| training data | human-only |
| input representation | A4: question + answer + metric + rubric + metadata |
| head | CORAL-style cumulative ordinal head |
| loss | EduRisk Ordinal Loss |
| tau | 0.7 |
| alpha_risk | 0.3 |
| beta_bce | 0.5 |
| lambda_LH | 2.0 |
| lambda_HL | 0.5 |
| class_balance_beta | 0.99 |
| normalized_cost | true |
| decode_primary | cumulative |
| decode_secondary | argmax_q |
| synthetic data | no |
| test tuning | no |

## Primary Metrics

| metric | reason |
| --- | --- |
| low_to_high_rate | primary low-score overestimation risk |
| MAE | overall ordinal accuracy |
| QWK | agreement under ordered labels |
| Kendall tau | rank correlation |
| Acc@1 | exact accuracy for true score 1 |
| Acc@2 | exact accuracy for true score 2 |
| Acc@5 | guard against collapsing high-score predictions |
| low_signed_bias | checks whether low scores are still overestimated |
| high_to_mid_or_low_rate | guard against over-suppressing high scores |
| monotonic_violation_rate | must remain zero under CORAL-style head |
| ExpectedEduRisk | diagnostic metric aligned to the new loss |

## First-run Success Criteria

| criterion | target |
| --- | --- |
| monotonic_violation_rate | 0 |
| low_to_high_rate | lower than QD-B1 |
| MAE | no more than QD-B1 + 0.02 |
| QWK | no less than QD-B1 - 0.03 |
| Acc@5 | no less than QD-B1 - 0.05 |
| high_to_mid_or_low_rate | no more than QD-B1 + 0.03 |
| low_signed_bias | closer to zero than QD-R1 |
| stretch target | low_to_high_rate <= 0.40 |

## Ablation Gate

Ablations should run only if `QD-ER1_EduRisk_human_only` completes cleanly and shows either:

- lower `low_to_high_rate` than QD-B1 without large MAE/QWK/Acc@5 regression, or
- promising dev-set risk reduction with clear failure analysis.

Do not launch ablations before the main run is reviewed.

## Planned Ablations

| run_id | change | question answered |
| --- | --- | --- |
| QD-ER1A_no_risk_cost | set alpha=0 | Is the explicit education-risk cost necessary? |
| QD-ER1B_no_class_balance | set w_y=1 | Is effective-number balancing useful beyond risk cost? |
| QD-ER1C_no_soft_target | replace soft CE with hard CE over q | Does soft ordinal encoding help? |
| QD-ER1D_no_cumBCE | set beta_bce=0 | Does cumulative BCE stabilize the CORAL head? |

## Optional Future Extension

CLOC-style ordinal contrastive learning with multi-margin loss can be considered after the EduRisk
main and ablation runs. It should be treated as a separate method because it changes representation
learning, not only the ordinal prediction loss.

## Reporting Tables To Add In Code Stage

These are future output targets, not created by this planning task:

| table | content |
| --- | --- |
| exp08_main_comparison.csv | QD-B0/QD-B1/QD-R1/QD-ER1 metrics |
| exp08_low_score_comparison.csv | low-label exact, bias, MAE, low_to_high |
| exp08_high_score_guardrail.csv | Acc@5 and high_to_mid_or_low |
| exp08_expected_risk.csv | ExpectedEduRisk overall and by label group |
| cumulative_decoding_vs_argmax_decoding.csv | cumulative threshold vs q-argmax diagnostic metrics |
| exp08_loss_component_scale.csv | L_total and component scales for train/dev |
| exp08_ablation_summary.csv | only if ablations run |

## Decision Logic

If QD-ER1 beats QD-B1 on low_to_high while preserving MAE/QWK/Acc@5, it becomes the primary
full-coverage training-method candidate. Exp7-C remains the human-in-the-loop fallback.

If QD-ER1 reduces low_to_high but hurts Acc@5 or QWK too much, report it as a useful diagnostic loss
but not as the final automatic scorer.

If QD-ER1 fails to improve low_to_high, use the ablation gate to identify whether the risk term,
soft target, or class balance is responsible before proposing a second version.
