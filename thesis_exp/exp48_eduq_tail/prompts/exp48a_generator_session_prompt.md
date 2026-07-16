# Exp48A generator session

You are the generator for a train-only qualification pilot. Read the private JSONL blueprint packet supplied with this prompt and produce exactly one JSONL family row per blueprint. Do not use or request dev/test data, labels, teacher scores, or historical predictions.

For each blueprint, create a materially new sibling question in the same language, metric, and education setting. It must change the substantive content, entities, or scenario rather than paraphrase the source. Then define exactly four atomic, directly verifiable criteria with IDs and types fixed as follows: `E1` essential, `E2` essential, `S1` supporting, and `P1` prohibited.

Use only the locked program `eduq_tail_v1`:

- score 2: a listed major essential criterion is `violated`, while at least one required essential component remains `satisfied` or `partial`; or a prohibited condition is violated;
- score 5: every required essential criterion is `satisfied`, no prohibited condition is violated, and the configured number of supporting criteria is satisfied;
- score 3: every other scorable criterion-state pattern.

Emit this exact executable program: `score5_required_essential_ids=[E1,E2]`, `score5_min_supporting_satisfied=1`, `score2_major_omission_ids=[E1]`, and `prohibited_ids=[P1]`. Emit these exact intended state templates: score 2 = `{E1: violated, E2: satisfied, S1: partial, P1: satisfied}`; score 3 = `{E1: partial, E2: satisfied, S1: partial, P1: satisfied}`; score 5 = `{E1: satisfied, E2: satisfied, S1: satisfied, P1: satisfied}`. The answer text must genuinely realize its template.

State semantics for a `prohibited` criterion are strict: `satisfied` means the answer successfully avoids the prohibited behavior; `violated` means the prohibited behavior occurs. Never mark an answer `violated` merely because it correctly avoids the behavior.

Generate three answers in the same request/session, intended for scores 2, 3, and 5. Match their length and surface style deliberately: for Chinese, write every answer as six complete sentences and 180-220 normalized characters; for English, write every answer as six complete sentences and 100-130 words. In every case, maximum/minimum normalized character length must be at most 1.5. The score-2 answer must use its full length for relevant peripheral context while omitting E1; it must not be shorter, broken, or nonsensical. A lower-quality answer must remain fluent and similarly detailed in surface form; its weakness must come from criterion content, not brevity, broken grammar, or obvious markers. Do not mention target scores or encode score hints in answer text. Record private criterion states for every answer. The states must reproduce the intended score under the locked program.

Return JSONL only, conforming to `schemas/exp48a_synthetic_family_schema.json`. Preserve `source_blueprint_id`, assign unique `family_id`, `synthetic_question_key`, and `answer_id` values, and include `generator_provenance.model_family` plus a unique `session_id`. Do not include human/model labels from the source packet.
