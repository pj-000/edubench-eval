# Exp48A generator session

You are the generator for a train-only qualification pilot. Read the private JSONL blueprint packet supplied with this prompt and produce exactly one JSONL family row per blueprint. Do not use or request dev/test data, labels, teacher scores, or historical predictions.

For each blueprint, create a materially new sibling question in the same language, metric, and education setting. It must change the substantive content, entities, or scenario rather than paraphrase the source. Then define 4-6 atomic, directly verifiable criteria. Use at least two `essential` criteria; optional criteria may be `supporting` or `prohibited`.

Use only the locked program `eduq_tail_v1`:

- score 2: a listed major essential criterion is `violated`, while at least one required essential component remains `satisfied` or `partial`; or a prohibited condition is violated;
- score 5: every required essential criterion is `satisfied`, no prohibited condition is violated, and the configured number of supporting criteria is satisfied;
- score 3: every other scorable criterion-state pattern.

Generate three answers in the same request/session, intended for scores 2, 3, and 5. Keep language and style comparable and ensure maximum/minimum normalized character length is at most 1.5. Do not mention target scores or encode score hints in answer text. Record private criterion states for every answer. The states must reproduce the intended score under the locked program.

Return JSONL only, conforming to `schemas/exp48a_synthetic_family_schema.json`. Preserve `source_blueprint_id`, assign unique `family_id`, `synthetic_question_key`, and `answer_id` values, and include `generator_provenance.model_family` plus a unique `session_id`. Do not include human/model labels from the source packet.
