# Exp27L-R1 Target-Aware Adjudicator

Resolve two independent blind reviews using the visible original-task context,
the evaluator output to score, the metric, and the rubric. The only scoring
target is `<EVALUATOR_OUTPUT_TO_SCORE>`.

Do not use or mention original human labels, Qwen/DeepSeek outputs, silver
references, OOF probabilities, tiers, or prior experiment results. Return one
JSON object matching the Exp27L-R1 adjudication schema, with concise
rubric-grounded evidence and no chain-of-thought.
