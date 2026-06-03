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
| A0 | A0_answer_only | completed | 0.6668169522091975 | 0.5408776675683798 | 0.3507460708301964 | 0.8446601941747572 | 280.93081647398844 | 0.0 |
| A1 | A1_question_answer | completed | 0.6492335437330928 | 0.5405770964833183 | 0.3699011680295815 | 0.7864077669902912 | 355.8553106936416 | 0.0 |
| A2 | A2_question_answer_metric | reused_exp02 | 0.7299368800721371 | 0.4238052299368801 | 0.5692855146538734 | 0.5339805825242718 | 370.06123554913296 | 0.0 |
| A3 | A3_question_answer_metric_rubric | completed | 0.7168620378719567 | 0.4236549443943492 | 0.5914073850965026 | 0.39805825242718446 | 477.8860187861272 | 0.0 |
| A4 | A4_question_answer_metric_rubric_metadata | completed | 0.7412082957619477 | 0.4030658250676285 | 0.5940582341762678 | 0.44660194174757284 | 502.32947976878614 | 0.0 |

## 训练前硬化检查
- rubric mode: corrected
- rubric quality audit: PASS
- zh Scenario Element Integration vs Instruction Following & Task Completion: PASS
- human confirmation needed: NO
- 正式训练前需要先完成服务器 smoke test，并人工审阅 rubric quality audit 的 ERROR/WARNING。

## 后续训练计划
第一轮正式训练只跑 A3 和 A4；A0/A1 资源不足时后补，A2 默认不重训。
