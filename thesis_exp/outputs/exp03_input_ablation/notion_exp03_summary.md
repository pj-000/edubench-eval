# Exp3 输入信息消融实验总结

## 实验目的
Exp3 用来验证 rubric 和 metadata 是否能改善教育评分模型与人类评分的一致性，尤其关注低分识别。

## 输入模板
- A0: answer only
- A1: question + answer
- A2: question + answer + metric，复用 Exp2 CE baseline
- A3: question + answer + metric + rubric
- A4: question + answer + metric + rubric + scenario/subject/education_level/language

## 数据
固定使用 Exp0.1 paper_like_triple_seed42 split：train=2654，dev=664，test=2218。

## 当前状态
| ablation_id | template_name | status | test_accuracy | test_MAE_label | test_kendall_tau | test_low_to_high_rate | mean_token_length | truncation_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | A0_answer_only | pending |  |  |  |  | 280.93081647398844 | 0.0 |
| A1 | A1_question_answer | pending |  |  |  |  | 355.8553106936416 | 0.0 |
| A2 | A2_question_answer_metric | reused_exp02 | 0.7299368800721371 | 0.4238052299368801 | 0.5692855146538734 | 0.5339805825242718 | 370.06123554913296 | 0.0 |
| A3 | A3_question_answer_metric_rubric | pending |  |  |  |  | 478.0715317919075 | 0.0 |
| A4 | A4_question_answer_metric_rubric_metadata | pending |  |  |  |  | 502.51318641618496 | 0.0 |

## 重点审阅项
- A2 是否正确复用 Exp2
- A3/A4 rubric 是否非空
- A4 是否没有引入 generator_model / answer_model / human score
- low_to_high_rate 是否下降
- 加入 rubric 后是否产生明显截断

## 后续训练计划
第一轮正式训练只跑 A3 和 A4；A0/A1 资源不足时后补，A2 默认不重训。
