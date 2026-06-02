# Notion Exp02 Paper Notes

Exp2 should be described as a baseline experiment rather than a proposed new
method. The key contribution is diagnostic: a compact supervised CE scorer can
match many overall evaluator metrics, while still failing on low-score cases.

Paper-ready points:

- The baseline input intentionally excludes rubrics and metadata.
- Qwen3-Reranker-0.6B is used as a sequence-classification model.
- Test Accuracy is 0.7299.
- Test MAE_label is 0.4238.
- Test Signed Bias is +0.1410.
- Test Kendall tau is 0.5693.
- Test low_to_high_rate is 0.5340.

Motivation for future experiments:

- Exp3: add rubric or metadata context.
- Exp4: test target variants.
- Exp5: reduce low-score overestimation with loss or sampling changes.
- Exp6: examine robustness across task slices.
- Exp7: calibrate probabilities using saved logits and probabilities.
