# Codex Independent Exp17-D1 Hidden Failure Audit 中文报告

## 范围与约束

- 本报告只审阅 dev D1 manual-review 文件中的 27 个 label2 but predicted 4/5 case。
- 未训练模型，未加载 transformers，未读取 test，未生成 checkpoint。
- 未生成 raw predictions/jsonl/npy/npz/log；仅生成独立标注 CSV、summarize 输出和本中文报告。
- 标注原则：不能从 question / answer / rubric 找到明确低分依据时，标 `possible_label_conflict` 或 `review_only`，不把模型高分当作 primary failure。

## 总体结论

- 总 case 数：27；annotation validation issues：0。
- rubric-linked hidden failure rate：0.4074。
- possible label conflict rate：0.5556。
- strong_or_weak_train_signal_rate：0.1481。
- max question group rate：0.7407，样本高度集中在同一营销经理纠错 question group。
- summarize 默认建议进入 Exp17-A：`false`。
- summarize reason：rubric_linked_hidden_failure_rate < 0.60; possible_label_conflict_rate > 0.35; strong_or_weak_train_signal_rate < 0.50; WARNING: Failure cases are highly concentrated in one question group; Exp17-A must use train-side weak labels and should avoid question_key-specific features.

## Failure Mode 数量

- `format_violation`: 1 / 27，rate=0.0370，rubric_linked=1，hidden=0，conflict=0
- `surface_fluent_but_hidden_defect`: 4 / 27，rate=0.1481，rubric_linked=4，hidden=4，conflict=0
- `insufficient_evidence`: 7 / 27，rate=0.2593，rubric_linked=7，hidden=7，conflict=0
- `possible_label_conflict`: 15 / 27，rate=0.5556，rubric_linked=0，hidden=0，conflict=15

## Trainability 数量

- `weak_train_signal`: 4 / 27，rate=0.1481
- `format_auxiliary_signal`: 1 / 27，rate=0.0370
- `pairwise_only`: 7 / 27，rate=0.2593
- `review_only`: 15 / 27，rate=0.5556

## 27 个 Case 的分组结论

### A. 可 evidence_positive 的场景整合缺陷

这些答案表面上完成了纠错任务，但在 `Scenario Element Integration` 指标下只有通用纠错文本，没有个性化或场景元素整合；rubric 明确把这种情况描述为表层引用/弱上下文连接。

- Case 21 `6304a049e36db27cde8ac17418f406f0b8d94687` - Scenario Element Integration；primary=`surface_fluent_but_hidden_defect`；notes：Scenario integration rubric expects key scenario elements/personalization; answer is a generic correction with no personalized or contextual adaptation beyond the original answer.
- Case 24 `bb99da8830bc11b1828c4a9637d65ecb7ee71420` - Scenario Element Integration；primary=`surface_fluent_but_hidden_defect`；notes：Scenario integration rubric expects key scenario elements/personalization; answer is a generic correction with no personalized or contextual adaptation beyond the original answer.
- Case 25 `ab6c4ba7b5065f647c29431814feeb3a00e9c7b0` - Scenario Element Integration；primary=`surface_fluent_but_hidden_defect`；notes：Scenario integration rubric expects key scenario elements/personalization; answer is a generic correction with no personalized or contextual adaptation beyond the original answer.
- Case 26 `fceb9659e6f4b3b5648c977ec612b79ae8a6ae29` - Scenario Element Integration；primary=`surface_fluent_but_hidden_defect`；notes：Scenario integration rubric expects key scenario elements/personalization; answer is a generic correction with no personalized or contextual adaptation beyond the original answer.

### B. 只能 pairwise_low 的弱信号

这些答案可见的缺陷主要是启发性、推理深度或解释充分性不足；缺陷和 rubric 有关系，但严重程度不足以把它们当作直接 evidence-positive。

- Case 14 `a7f69c77078e9926635bb2b77ebdd28d5733ae36` - Clarity, Simplicity & Inspiration；primary=`insufficient_evidence`；notes：Clarity/inspiration rubric rewards concise, accessible, inspiring delivery; answer is clear but generic corrective prose with little inspirational framing. I would not use this as direct evidence-positive because the observed weakness is mild relative to a label-2 score.
- Case 16 `1bdcdd67d27147705080e74bea8e7252e9ce95bb` - Clarity, Simplicity & Inspiration；primary=`insufficient_evidence`；notes：Clarity/inspiration rubric rewards accessible, thought-provoking delivery; answer is generic and only minimally learner-oriented. I would not use this as direct evidence-positive because the observed weakness is mild relative to a label-2 score.
- Case 17 `c8f82a756a23aa9b5e70f52f867ac392e2aec089` - Clarity, Simplicity & Inspiration；primary=`insufficient_evidence`；notes：Clarity/inspiration rubric rewards accessible, inspiring delivery; answer is coherent but generic and not especially inspiring. I would not use this as direct evidence-positive because the observed weakness is mild relative to a label-2 score.
- Case 18 `e9b358073140893a17f069d26d8fe2adc338f088` - Clarity, Simplicity & Inspiration；primary=`insufficient_evidence`；notes：Clarity/inspiration rubric rewards accessible, inspiring delivery; answer is concise but generic and offers little inspirational framing. I would not use this as direct evidence-positive because the observed weakness is mild relative to a label-2 score.
- Case 19 `abd894527f401792d7a5cb4438ee15420de3ea1d` - Reasoning Process Rigor；primary=`insufficient_evidence`；notes：Reasoning rigor rubric rewards complete, rigorous steps; answer explains the correction but reasoning is brief and mostly assertive. I would not use this as direct evidence-positive because the observed weakness is mild relative to a label-2 score.
- Case 20 `b6b6dda03618d53072e9cb795d23ca1a8dc79a57` - Reasoning Process Rigor；primary=`insufficient_evidence`；notes：Reasoning rigor rubric rewards complete, rigorous steps; answer gives a plausible correction but limited step-by-step justification. I would not use this as direct evidence-positive because the observed weakness is mild relative to a label-2 score.
- Case 22 `057fb10dadb0ef587ef63e8c0692fb069e747f33` - Reasoning Process Rigor；primary=`insufficient_evidence`；notes：Reasoning rigor rubric rewards complete, rigorous steps; answer gives a reasonable explanation but limited argumentation depth. I would not use this as direct evidence-positive because the observed weakness is mild relative to a label-2 score.

### C. format_auxiliary

该样本在 JSON 后追加了非 JSON Explanation，能作为格式/任务约束辅助信号，但不建议作为 hidden-failure evidence-positive。

- Case 27 `46e715a2df1e392b0e0ddb3fee67f5a339e5e608` - Motivation, Guidance & Positive Feedback；primary=`format_violation`；notes：Answer provides a JSON block but then appends an Explanation section outside JSON, violating the explicit JSON-only task format; motivation-score low label itself remains only weakly supported.

### D. exclude/review_only

这些样本无法仅凭 question/answer/rubric 可靠解释 label=2，或需要外部标准答案/事实 key；保留质检，不进入训练信号。

- Case 1 `f93c70f6bc17688bb1aa72c33100e1716657dc59` - Content Relevance & Scope Control；primary=`possible_label_conflict`；notes：Content/relevance low score appears to depend on an external answer key or historical interpretation; answer is on-topic and self-consistent.
- Case 2 `ec3c258a92d9ea146d07d5b2d54170aa37b10793` - Basic Factual Accuracy；primary=`possible_label_conflict`；notes：Factual low score would require verifying whether A or D is the expected key; text alone does not prove several/key inaccuracies.
- Case 3 `308181498ec49e2d421c165fdf652ef8991f34f9` - Instruction Following & Task Completion；primary=`possible_label_conflict`；notes：Corrected answer and explanation satisfy the requested error-correction task; no visible major instruction or format failure.
- Case 4 `a9fb1437d7ef739f816e75d995054de210ca96a8` - Instruction Following & Task Completion；primary=`possible_label_conflict`；notes：Corrected answer and explanation are valid JSON and address the original error; no visible major instruction failure.
- Case 5 `f45ee7ba1b382a63c9d4d3f14889365d54dc1cbd` - Error Identification & Correction Precision；primary=`possible_label_conflict`；notes：Error is identified and corrected accurately; no critical omission or false positive is visible.
- Case 6 `60084a95316f0f87040c9c1c0c24a16be077aa06` - Instruction Following & Task Completion；primary=`possible_label_conflict`；notes：Corrected answer and explanation complete the task; no visible reason for label 2 under instruction-following rubric.
- Case 7 `6b1d22832f8c2d9e51459ade117c52581ec16bf8` - Instruction Following & Task Completion；primary=`possible_label_conflict`；notes：Corrected answer and explanation complete the task; no visible reason for label 2 under instruction-following rubric.
- Case 8 `02257ff2c77501987e43989cc8c71a017fb39dad` - Instruction Following & Task Completion；primary=`possible_label_conflict`；notes：Plain JSON object is valid and task-compliant; no visible reason for label 2 under instruction-following rubric.
- Case 9 `760fc72a43fcce0f05f6e57f16c28416ee7a2536` - Error Identification & Correction Precision；primary=`possible_label_conflict`；notes：Error is identified and corrected accurately; no critical omission or false positive is visible.
- Case 10 `3783f563820e72434b75d23f81fd58fb010931f7` - Content Relevance & Scope Control；primary=`possible_label_conflict`；notes：Short answer is directly responsive; low relevance score would require external key/context not present in question-answer-rubric.
- Case 11 `d827928fc39a06825d345c6f8c0685eba89b65cd` - Error Identification & Correction Precision；primary=`possible_label_conflict`；notes：Error is identified and corrected accurately; no critical omission or false positive is visible.
- Case 12 `9ab1bf35a1c05a0d05b45d75620e919312780063` - Instruction Following & Task Completion；primary=`possible_label_conflict`；notes：JSON contains an Answer field and addresses the requested question; low instruction-following score is not explained by visible format/task failure.
- Case 13 `28cb100bda849f499ad3ad11500b7d1b494be4a7` - Error Identification & Correction Precision；primary=`possible_label_conflict`；notes：Error is identified and corrected accurately; no critical omission or false positive is visible.
- Case 15 `a30876aee5bfcd792a475f60264f22dfc7b74b91` - Basic Factual Accuracy；primary=`possible_label_conflict`；notes：Short answer factuality low score would require external key verification; no intrinsic factual error is visible from the provided text.
- Case 23 `04bfbf54a7eeef628f3799e10e300520dcee17ca` - Instruction Following & Task Completion；primary=`possible_label_conflict`；notes：Automatic-grading answer follows requested JSON fields and gives detailed feedback; low instruction-following label is not explained without an external grading key or stricter scoring standard.

## Question Group / Metric 分组结论

- `14ba3cb00f998348fe1c491eab066379d3bf192b` / Instruction Following & Task Completion: n=5，主类=`possible_label_conflict`，rubric_linked_rate=0.0000，hidden_rate=0.0000，conflict_rate=1.0000，建议=`review_or_downweight`
- `14ba3cb00f998348fe1c491eab066379d3bf192b` / Clarity, Simplicity & Inspiration: n=4，主类=`insufficient_evidence`，rubric_linked_rate=1.0000，hidden_rate=1.0000，conflict_rate=0.0000，建议=`pairwise_or_review`
- `14ba3cb00f998348fe1c491eab066379d3bf192b` / Error Identification & Correction Precision: n=4，主类=`possible_label_conflict`，rubric_linked_rate=0.0000，hidden_rate=0.0000，conflict_rate=1.0000，建议=`review_or_downweight`
- `14ba3cb00f998348fe1c491eab066379d3bf192b` / Scenario Element Integration: n=4，主类=`surface_fluent_but_hidden_defect`，rubric_linked_rate=1.0000，hidden_rate=1.0000，conflict_rate=0.0000，建议=`evidence_positive_candidate`
- `14ba3cb00f998348fe1c491eab066379d3bf192b` / Reasoning Process Rigor: n=3，主类=`insufficient_evidence`，rubric_linked_rate=1.0000，hidden_rate=1.0000，conflict_rate=0.0000，建议=`pairwise_or_review`
- `9d1179a873f8e7454e4075453ea96fee9e73ecff` / Basic Factual Accuracy: n=2，主类=`possible_label_conflict`，rubric_linked_rate=0.0000，hidden_rate=0.0000，conflict_rate=1.0000，建议=`review_or_downweight`
- `9d1179a873f8e7454e4075453ea96fee9e73ecff` / Content Relevance & Scope Control: n=2，主类=`possible_label_conflict`，rubric_linked_rate=0.0000，hidden_rate=0.0000，conflict_rate=1.0000，建议=`review_or_downweight`
- `1bbfb9a5f532b875aaa1b5a1500fb88535b21a51` / Motivation, Guidance & Positive Feedback: n=1，主类=`format_violation`，rubric_linked_rate=1.0000，hidden_rate=0.0000，conflict_rate=0.0000，建议=`format_auxiliary_candidate`
- `9d1179a873f8e7454e4075453ea96fee9e73ecff` / Instruction Following & Task Completion: n=1，主类=`possible_label_conflict`，rubric_linked_rate=0.0000，hidden_rate=0.0000，conflict_rate=1.0000，建议=`review_or_downweight`
- `9dcad11d15cc245e5dabc70ec2358208f1139f70` / Instruction Following & Task Completion: n=1，主类=`possible_label_conflict`，rubric_linked_rate=0.0000，hidden_rate=0.0000，conflict_rate=1.0000，建议=`review_or_downweight`

## 是否建议进入 Exp17-A evidence head

不建议直接进入 Exp17-A evidence head。原因是可作为 evidence-positive 的样本只有 4/27，且全部来自同一 question group 的 `Scenario Element Integration` 指标；同时 15/27 被标为 possible label conflict / review-only，整体冲突率超过 summarize 阈值。若直接训练 evidence head，模型很容易学习 question-key 或 metric-group 偏差，而不是可复用的 hidden-failure 证据。

## 下一步建议

1. 先复核 History/Annales 题的标准答案 key，尤其 A Marc Bloch 与 D Fernand Braudel 的预期答案差异；没有 key 前不要把这些样本作为事实错误 evidence。
2. 对营销经理纠错 question group 重新抽样，避免单一 question group 占 74% 以上；优先寻找不同学科、不同任务类型的 hidden failure。
3. 把 4 个 `Scenario Element Integration` 样本暂作为弱 evidence-positive 候选，但需要 train-side 扩展后再决定是否训练。
4. 把 7 个 `pairwise_low` 只用于 matched hard-negative / pairwise 排序实验，不作为直接 hidden-failure 正例。
5. 把 Case 27 单独进入 format/task auxiliary 池；它不是 evidence head 的主信号。
