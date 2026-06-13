# QD-PR1 Pairwise Formal Result Diagnosis

## Scope

This diagnosis reads existing QD-PR1 raw run outputs, summary tables, predictions, pairwise
diagnostics, and loss/debug history. It does not train a model, call an API, generate synthetic
data, modify raw predictions/arrays/logs, submit checkpoint files, or start QD-PR2.

## Required Answers

- Did QD-PR1 reduce low-to-high? **NO.** Test low_to_high is `0.5161`, worse than QD-B1 `0.4516`; delta vs QD-B1 is `0.0645`.
- Did QD-PR1 beat QD-B1? **NO.** MAE delta `0.0266`, QWK delta `-0.0539`, and Acc@5 delta `-0.0503` are all unfavorable.
- Did pairwise training damage ordinal monotonicity? **YES.** Test monotonic violation is `0.7616`; delta vs QD-B1 is `0.4497`.
- Did pairwise training improve pairwise score gaps on dev? **YES, partially.** Dev pair accuracy is `0.7964`, but only `0.4946` of pairs satisfy the configured margin.
- Are low-high pairs better separated after training? **Partially.** Dev low-high pair accuracy is `0.7920`, mean score gap `1.0548`, but margin satisfaction is only `0.4040`.
- Is pointwise calibration worse? **YES.** The model improves pair separation while worsening test MAE, QWK, Acc@5, and monotonicity relative to QD-B1.
- Does label=2 remain problematic? **YES.** On test label=2, low_to_high is `0.6818` and monotonic violation is `0.4545`.
- Should QD-PR2 start? **Recommended only as a controlled anchored fine-tuning experiment, not as an unanchored from-scratch rerun.**

## Main Test Metrics

| model | low_to_high | MAE_label | QWK | Acc@5 | monotonic_violation |
| --- | ---: | ---: | ---: | ---: | ---: |
| QD-B1 raw | 0.4516 | 0.4279 | 0.6012 | 0.7419 | 0.3119 |
| QD-PR1 formal | 0.5161 | 0.4545 | 0.5473 | 0.6916 | 0.7616 |

## Prediction Distribution

Test pred_label distribution is `1: 9 (0.0082), 2: 0 (0.0000), 3: 63 (0.0571), 4: 502 (0.4551), 5:
529 (0.4796)`. The model still concentrates predictions in high labels, while label=1/2 cases are
frequently promoted into 4/5.

## Low-score Failure Pattern

Across test true labels 1-2, low_to_high is `0.5161` (16/31). Label=2 is the key failure:
low_to_high `0.6818`, mean predicted label `3.8182`, signed error `1.8182`.

## Pairwise vs Pointwise Diagnostics

Loss history supports a split between pairwise separation and pointwise calibration. In the first
100 steps, mean score gap is `0.0166`, L_point `0.6564`, and L_pair `1.7975`. In the last 100 steps,
mean score gap rises to `2.0469`, L_point falls to `0.0008`, and L_pair remains `0.3224`. This is
consistent with pairwise ordering being optimized without preserving calibrated cumulative
probabilities.

## Interpretation

QD-PR1 is a negative formal result. The pairwise objective learns useful ranking/gap signals on dev
pairs, but the independent ordinal head is not anchored strongly enough to preserve valid cumulative
behavior. The result should not be presented as an effective method. It should be presented as
evidence that pairwise supervision must be constrained by a stronger pointwise/monotonic anchor.

## QD-PR2 Recommendation

QD-PR2 is recommended only with controlled changes:

- Initialize from the QD-B1 checkpoint instead of training from scratch.
- Sweep `lambda_pair` only in `{0.05, 0.1}`.
- Add monotonic regularization or use a rank-consistent head so pairwise gaps cannot break cumulative order.
- Use high-comparability pairs only, prioritizing `same_question` and then `same_metric_language`.
- Fine-tune for 2-3 epochs rather than running a full from-scratch 10-epoch training job.

Do not start QD-PR2 from this script; this report only proposes the controlled follow-up.
