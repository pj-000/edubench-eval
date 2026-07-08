You are auditing whether the original human score is reliable.

You will receive:
1. the original question/answer/rubric input,
2. your previous blind scoring result,
3. the original human score.

Return exactly one JSON object matching the provided schema. Keep the blind
fields unchanged unless they are invalid JSON or violate the schema. In audit,
decide whether the original human score is reliable, adjacent/plausible, in
conflict with the blind judgment, or unclear.

Do not treat the teacher score as gold. The teacher is an auditor, not a
replacement for human labels.
