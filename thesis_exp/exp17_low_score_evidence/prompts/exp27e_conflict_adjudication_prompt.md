# Exp27E Conflict Adjudication Prompt

You are adjudicating one EduBench educational scoring case. The case comes from
the train split only. Your job is not to replace all human labels, but to
resolve high-value conflicts between two independent teacher audits.

Use only the provided question, answer, metric, rubric, metadata, original train
score, and compact teacher disagreement summary. Do not assume either teacher is
always correct. Ground every judgement in the rubric.

Return only valid JSON following the provided schema.

Required reasoning policy:

1. Decide whether the original human score is plausible under the rubric.
2. Decide whether the Qwen teacher judgement is plausible.
3. Decide whether the DeepSeek teacher judgement is plausible.
4. If the case is ambiguous, return a score range and mark review-only or
   low-weight rather than forcing a false precise label.
5. Do not produce hidden chain-of-thought. The `adjudication_reason` should be a
   concise, evidence-based explanation suitable for dataset documentation.

Important definitions:

- `hidden_or_missing_failure`: the answer looks fluent or plausible but misses a
  required rubric element, task constraint, personalization requirement, or
  factual/rubric-critical point.
- `visible_failure`: the defect is obvious from the answer surface, such as
  irrelevant content, formatting failure, or clearly incomplete answer.
- `no_failure`: no major rubric-grounded defect is found.
- `unclear`: the case cannot be confidently judged from the available
  question/rubric/answer context.

Training-use guidance:

- `high_weight`: use when the adjudicated score and failure evidence are clear.
- `low_weight`: use when the case is broadly usable but has adjacent-score
  ambiguity.
- `review_only`: use when teacher disagreement or original-label uncertainty is
  substantial.
- `exclude`: use when the case appears mislabeled, under-specified, or impossible
  to resolve without external information.
