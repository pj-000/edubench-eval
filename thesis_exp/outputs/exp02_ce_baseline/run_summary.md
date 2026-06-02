# Exp2 CE Baseline Run Summary

Model: `Qwen3-Reranker-0.6B`

Training objective: 5-class cross-entropy sequence classification.

Input template: question + answer + metric only.

Best checkpoint:
`thesis_exp/artifacts/exp02_ce_baseline/checkpoints/edubench_evaluator_0_6b_ce/best`

| split | Accuracy | MAE_label | MAE_expected | Signed Bias | Kendall tau |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev | 0.7108 | 0.3976 | 0.3682 | 0.1155 | 0.5718 |
| test | 0.7299 | 0.4238 | 0.3865 | 0.1410 | 0.5693 |

The formal run completed at `global_step=210`.
