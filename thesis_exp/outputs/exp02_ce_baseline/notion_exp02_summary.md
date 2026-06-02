# Notion Exp02 Summary

Exp2 trains a Qwen3-Reranker-0.6B five-class CE baseline on the locked split.

Protocol:

- Input: question + answer + metric only.
- Output: score class 1 to 5.
- Training: full fine-tuning sequence classification.
- Formal test rows: 2218.

Headline result:

- Accuracy: 0.7299
- MAE_label: 0.4238
- Signed Bias: +0.1410
- Kendall tau: 0.5693
- low_to_high_rate: 0.5340

Takeaway:

The model is close to Exp1 overall, but the low-score blind spot remains severe.
