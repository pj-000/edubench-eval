# Exp2 Formal Result Report

Exp2 is a cross-entropy baseline, not an innovation method. Its purpose is to
test whether a small supervised scorer can reproduce human-aligned EduBench
quality scores under a deliberately simple protocol.

The input is question + answer + metric only. Rubrics, scenario metadata,
subject metadata, education level, language, generator identity, score anchors,
and chain-of-thought instructions are excluded from the model text.

The model is `Qwen3-Reranker-0.6B` used as a sequence-classification backbone
with five labels for scores 1 to 5.

On the locked test split, Exp2 reaches Accuracy=0.7299,
MAE_label=0.4238, Signed Bias=+0.1410,
and Kendall tau=0.5693. The expected-score MAE is
0.3865, and Within-1 Accuracy is
0.9531.

The key remaining failure mode is the low-score blind spot. The test
low_to_high_rate is 0.5340, meaning that more than half of
true low-score examples are still predicted as high scores. Low-score exact
match is only 0.2136, while Acc@5 is
0.8085. This pattern shows that the CE baseline is
strong overall but remains too generous on genuinely poor answers.

Compared with the PDF EduBenchEvaluator and the Exp1 reproduced
EduBenchEvaluator, the Exp2 CE baseline is close on overall metrics:

| system | MAE | Exact Match | Signed Bias | Kendall tau | low_to_high_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| PDF EduBenchEvaluator | 0.4300 | 0.7250 | 0.2460 | 0.5080 | NA |
| Exp1 reproduced EduBenchEvaluator | 0.4358 | 0.7241 | 0.2528 | 0.5052 | 0.5340 |
| Exp2 trained CE baseline | 0.4238 | 0.7299 | 0.1410 | 0.5693 | 0.5340 |

Overall, Exp2 shows that a compact supervised CE scorer can approach the
published and reproduced evaluator-level metrics. However, the low-score
overestimation pattern motivates the next experiments.

Exp3 should test rubric-aware or metadata-aware inputs. Exp4 can examine
alternative target transformations. Exp5 should focus on low-score-sensitive
losses or sampling. Exp6 can study robustness and domain slices. Exp7 should
use the saved logits/probabilities for calibration and threshold analysis.
