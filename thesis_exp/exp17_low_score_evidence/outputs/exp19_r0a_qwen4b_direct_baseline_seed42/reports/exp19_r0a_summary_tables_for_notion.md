# Exp19-R0A Summary Tables for Notion

## 1. Overall Metrics

| 评分器 | MAE↓ | Signed Bias | Exact Match↑ | Kendall τ↑ | low-to-high↓ | label2 recall↑ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B direct | 0.644 | 0.403 | 0.541 | 0.191 | 80.4% | 5.3% |
| EduBenchEvaluator | 0.383 | 0.254 | 0.706 | 0.489 | 61.3% | 18.6% |
| DeepSeek-V3 judge | 0.556 | 0.352 | 0.585 | 0.321 | 88.6% | 1.8% |
| DeepSeek-R1 judge | 0.568 | 0.232 | 0.570 | 0.321 | 66.0% | 17.7% |
| QwQ-plus judge | 0.561 | 0.307 | 0.586 | 0.314 | 76.8% | 13.6% |
| GPT-4o judge | 0.550 | 0.401 | 0.571 | 0.289 | 86.2% | 2.8% |

## 2. Low-Score Bias

| 评分器 | 人类均分 1-2 时的模型预测均值 | 低分高估率 | 是否低分高估 |
|---|---:|---:|---|
| Qwen3-4B direct | 4.402 | 80.4% | 极严重 |
| DeepSeek-V3 judge | 4.378 | 88.6% | 极严重 |
| GPT-4o judge | 4.340 | 86.2% | 极严重 |
| QwQ-plus judge | 4.196 | 76.8% | 极严重 |
| DeepSeek-R1 judge | 3.929 | 66.0% | 严重 |
| EduBenchEvaluator | 3.658 | 61.3% | 严重 |

## 3. Per-Label Accuracy

| 评分器 | 1分准确率 | 2分准确率 | 3分准确率 | 4分准确率 | 5分准确率 |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B direct | 3.5% | 5.3% | 9.5% | 14.6% | 90.9% |
| EduBenchEvaluator | 37.2% | 18.6% | 17.2% | 60.7% | 89.2% |
| DeepSeek-V3 judge | 6.0% | 1.8% | 5.6% | 31.1% | 89.0% |
| DeepSeek-R1 judge | 13.1% | 17.7% | 16.0% | 29.2% | 84.9% |
| QwQ-plus judge | 6.0% | 13.6% | 14.4% | 27.5% | 89.4% |
| GPT-4o judge | 0.0% | 2.8% | 5.8% | 25.9% | 89.4% |

## Key Takeaway

Qwen3-4B direct scoring parses perfectly, but strongly overestimates low-score samples: low-to-high is 80.4% and label2 recall is 5.3%. It is therefore a reference baseline, not a solved evaluator.
